# -*- coding: utf-8 -*-
"""
Northbridge Bank - Credit Risk Query Engine (Streamlit app)

Converted from Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb.
The verified query library, prompts, validation gate, retry/escalation flow and
pipeline orchestration are carried over from the notebook. Only the notebook-specific
parts (config.json loading, print/display output, test cells) were adapted for Streamlit.

Run locally:   streamlit run app.py
"""

import json
import os
import re
import sqlite3
from datetime import datetime

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

DB_PATH = "credit_risk_portfolio.db"
TEST_QUERIES_PATH = "test_queries.csv"      # optional - only used for sample questions
MODEL_NAME = "gpt-4o-mini"


# ----------------------------------------------------------------------------------
# Credentials (Streamlit secrets -> environment variables -> local config.json)
# ----------------------------------------------------------------------------------
def load_credentials():
    """Returns (api_key, api_base). api_base may be None (default OpenAI endpoint)."""
    api_key, api_base = None, None

    # 1. Streamlit secrets (Streamlit Community Cloud, or .streamlit/secrets.toml locally)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass  # no secrets file available

    # 2. Environment variables
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    # 3. Local config.json (same file the notebook used; do NOT commit it to GitHub)
    if not api_key and os.path.exists("config.json"):
        with open("config.json", "r") as file:
            config = json.load(file)
        api_key = config.get("OPENAI_API_KEY")
        api_base = api_base or config.get("OPENAI_API_BASE")

    return api_key, (api_base or None)


@st.cache_resource(show_spinner=False)
def get_llms(api_key, api_base):
    """Set up the LLM(s) with explicit credentials (same settings as the notebook)."""
    llm_ = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=api_key, base_url=api_base)
    evaluator_llm_ = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=api_key, base_url=api_base)
    return llm_, evaluator_llm_


@st.cache_resource(show_spinner=False)
def get_connection(db_path):
    """Read-only SQLite connection (URI mode with ?mode=ro prevents any write operation)."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()
if not OPENAI_API_KEY:
    st.error(
        "OpenAI credentials not found. Add OPENAI_API_KEY (and OPENAI_API_BASE, if you use a "
        "custom endpoint) to the app's Streamlit secrets, or set them as environment variables."
    )
    st.stop()

if not os.path.exists(DB_PATH):
    st.error(f"Database file '{DB_PATH}' not found. It must sit next to app.py in the repository.")
    st.stop()

llm, evaluator_llm = get_llms(OPENAI_API_KEY, OPENAI_API_BASE)
conn = get_connection(DB_PATH)


# ----------------------------------------------------------------------------------
# Database schema provided to the LLM in the prompts
# ----------------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ----------------------------------------------------------------------------------
# Verified Query Template Library (SQL copied unchanged from the notebook)
# ----------------------------------------------------------------------------------

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = '''
SELECT
    sector_master.sector_name,
    SUM(loan_master.total_outstanding) AS total_expsoure,
    SUM(Case
            WHEN loan_master.asset_classification IN ('Substandard', 'Doubtful', 'Loss' )
            THEN loan_master.total_outstanding
            ELSE 0
        END) AS npa_exposure
FROM sector_master
JOIN loan_master ON sector_master.sector_code = loan_master.sector_code
GROUP BY sector_master.sector_name
ORDER BY total_expsoure DESC
'''

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT loan_category,
       SUM(total_outstanding) AS total_outstanding,
       COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding DESC;
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_millions,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """

SELECT sector_master.sector_name,
       AVG(provisioning.provision_coverage_ratio) AS avg_coverage_ratio
FROM provisioning
JOIN loan_master ON provisioning.loan_account_number = loan_master.loan_account_number
JOIN sector_master ON loan_master.sector_code = sector_master.sector_code
WHERE provisioning.reporting_date = '2025-09-30'
GROUP BY sector_master.sector_name
ORDER BY avg_coverage_ratio DESC;

"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT
    borrower_name,
    sector_master.sector_name,
    loan_master.total_outstanding,
    loan_master.asset_classification AS total_outstanding
FROM loan_master
JOIN sector_master ON loan_master.sector_code = sector_master.sector_code
ORDER BY loan_master.total_outstanding DESC
LIMIT 10;

"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_millions DESC
LIMIT 5

"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
    loan_account_number,
    borrower_name,
    sector_master.sector_name,
    loan_master.total_outstanding,
    loan_master.days_past_due,
    loan_master.asset_classification
FROM loan_master
JOIN sector_master ON loan_master.sector_code = sector_master.sector_code
WHERE days_past_due > 0
ORDER BY days_past_due DESC;
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        ELSE 5
    END;
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
    borrower_ID,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30' and rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;

"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;

"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1.strip()
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2.strip().rstrip(';')
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3.strip().rstrip(';')
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4.strip().rstrip(';')
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5.strip().rstrip(';')
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6.strip().rstrip(';')
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7.strip().rstrip(';')
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8.strip().rstrip(';')
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9.strip().rstrip(';')
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10.strip().rstrip(';')
    }
}


# ----------------------------------------------------------------------------------
# Tool definitions
# ----------------------------------------------------------------------------------

# ---- Intent Classification Tool --------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

You are the intent router for a commercial bank's credit-risk query engine. Decide whether the user's question can be answered completely by ONE of the pre-approved verified query templates, or whether new SQL must be generated.

RULES
- Choose "verified" if a single template returns the metric(s), the breakdown and the time
    period the user is asking about. A template may return extra rows or columns (a later step
    filters them), but it must not be missing anything the user asked for.

- IMPORTANT: If a user asks about a specific sector (e.g., 'Real Estate') and a template like VQ1 provides the sector-wise breakdown for all sectors, select 'verified' with VQ1. Do not use 'generated' just because the user specified a single category or sector that is included in the template's output.

- Choose "generated" when the question needs a metric, filter, grouping or calculation that no
    template provides. Examples: restructured loans, interest rates, one IFRS 9 stage tracked over
    time, or any question that mixes ideas from several templates.

- A template that is similar is NOT a match. If not a match, choose "generated".

- Use only the query IDs listed below. Never invent one.


{user_question}

{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ---- Query Generation Tool -------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
You are a senior credit risk data analyst who writes SQLite SQL. Write one read-only SQL query that answers the question, using only the tables and columns in the schema.

RULES:
- Output ONLY the SQL query. No explanations, no markdown block wrappers.
- The query must start with SELECT or WITH.
- NPA / Impaired definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss').
- Latest reporting date is '2025-09-30' (use only for current/latest status queries).
- For tracking Stage 3 portfolio trends (like Stage 3 EAD and ECL) across multiple quarters, DO NOT filter to a single latest reporting date. Show the trend across all available reporting_date values in provisioning.
- For Stage 3 analysis, filter strictly on `ifrs9_stage = 3`. Do not apply any unnecessary filters for restructured loans or NPA unless specifically requested.

Example structure for Stage 3 Quarterly Trends:
SELECT
    reporting_date,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS stage3_ead_millions,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS stage3_ecl_millions
FROM provisioning
WHERE ifrs9_stage = 3
GROUP BY reporting_date
ORDER BY reporting_date ASC;

User Question:
{user_question}

Schema Context:
{schema_context}
"""
    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ---- Query Validation Tool -------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "CRITICAL: This SQL belongs to a pre-approved VERIFIED TEMPLATE (e.g. VQ1, VQ2, etc.). "
        "These templates intentionally return broad aggregations across all categories or sectors (like all industry sectors). "
        "The query MUST NOT be rejected for lacking a WHERE clause that filters to a single category/sector mentioned in the question. "
        "A separate narrative response stage will extract the specific category row from the complete result table. "
        "Ensure you mark 'verdict' as 'yes' if the template query matches the general metrics, even if it lacks a specific category filter."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

You are an independent SQL reviewer on a bank's credit risk analytics team.
Decide whether the SQL correctly answers the user's question.

Check:
-  Are the right tables and columns used?
-  Is the metric and aggregation right (SUM vs AVG vs COUNT)?
-  Are the filters right? NPA = asset_classification IN ('Substandard','Doubtful','Loss'); latest provisioning/rating date = '2025-09-30'; restructured = is_restructured = 1.
-  Are joins correct, with no accidental row duplication?
-  Is the time logic right (\"trend\" = all quarters in order; \"latest/current\" = 2025-09-30)?

{track_context}

User Question:
{user_question}

Candidate SQL:
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    # NOTE (changed from notebook): the notebook appended " LIMIT 0" directly to the SQL, which is a
    # syntax error for templates that already end in LIMIT (VQ5, VQ6) and made them always escalate.
    # Wrapping the query in a subquery performs the same column-count comparison for every template.
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({candidate_sql.rstrip().rstrip(';')}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ---- Retry Generation Tool -------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
You are a senior credit risk data analyst. The SQL below FAILED validation. Fix it so it answers the original question and passes validation.

RULES
- Keep the original intent of the question. Fix exactly what the validation error describes.

- Output ONLY the corrected SQL. No explanation, no markdown fences.

- SQLite syntax, read-only: start with SELECT or WITH, a single statement, only tables/columns in the schema.

- NPA = asset_classification IN ('Substandard','Doubtful','Loss'). Latest date = '2025-09-30'.

- Show money in millions with ROUND(x / 1e6, 2) and an alias ending in _millions.


User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ---- Query Execution Tool --------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ---- Response Generation Tool ----------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

You are a credit-risk analyst writing for a senior, non-technical business user. Answer the user's question using ONLY the query result below.

RULES
- Start with the direct answer in less than a paragraph, using exact figures from the data.

- Money columns ending in _millions are USD millions; write them like $1,195.12M.

- Mention only the rows and columns relevant to the question. The table may contain more than was asked.

- Add at most two short sentences of useful context or a caveat.

- Do not invent numbers, do not mention SQL or tables, and do not repeat the whole table.

- If the result is empty, say that no matching records were found.


{user_question}

{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ----------------------------------------------------------------------------------
# Pipeline orchestration (same stages and order as the notebook; print() -> trace list)
# ----------------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, log and trace.
    '''

    trace = []

    def say(message):
        if verbose:
            trace.append(message)

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    say(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, 'warnings': [], 'trace': trace, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        say(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, 'warnings': exec_result['warnings'], 'trace': trace, **log}


# ----------------------------------------------------------------------------------
# Streamlit UI helpers
# ----------------------------------------------------------------------------------
def unique_columns(df):
    """Display-only: Streamlit cannot render duplicate column names (VQ5 returns two
    columns called total_outstanding), so suffix repeats. The underlying data is unchanged."""
    seen, cols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    out = df.copy()
    out.columns = cols
    return out


@st.cache_data(show_spinner=False)
def load_sample_questions():
    """Sample questions from test_queries.csv if it is present in the repo."""
    if os.path.exists(TEST_QUERIES_PATH):
        try:
            return pd.read_csv(TEST_QUERIES_PATH)['User Query'].dropna().tolist()
        except Exception:
            pass
    return [
        "What is the total outstanding by loan category?",
        "What is the NPA exposure in the Real Estate sector?",
        "Show the aging profile of the portfolio by days past due.",
        "What is the average interest rate by sector?",
        "How has Stage 3 exposure and ECL moved across quarters?",
    ]


def format_confidence(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value) if value is not None else "n/a"


def render_result(result):
    escalated = result['escalated']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", str(result['route']).capitalize())
    c2.metric("Verified query ID", result['query_id'] or "—")
    c3.metric("Confidence", format_confidence(result['confidence']))
    c4.metric("Rows returned", result['row_count'] if result['row_count'] is not None else "—")

    if result.get('match_reason'):
        st.caption(f"Routing reason: {result['match_reason']}")
    if result['retry_used']:
        st.info("The first generated query failed validation; the answer below uses the retried query.")

    st.subheader("Answer")
    if escalated:
        st.warning(result['narrative'])
    else:
        st.write(result['narrative'])

    for w in result.get('warnings', []):
        st.warning(f"Result check: {w}")

    st.subheader("SQL query used")
    if escalated:
        st.caption("Not executed - last candidate SQL shown for analyst review.")
    st.code(result['executed_sql'] or result['candidate_sql'] or "", language="sql")

    st.subheader("Raw data returned")
    df = result['dataframe']
    if df is None:
        st.caption("No data - the query was escalated before execution.")
    else:
        shown = unique_columns(df)
        st.dataframe(shown)
        st.download_button(
            "Download result as CSV",
            shown.to_csv(index=False).encode("utf-8"),
            file_name="query_result.csv",
            mime="text/csv",
        )

    with st.expander("Validation gate result"):
        st.json(result['gate_result'])
    with st.expander("Pipeline trace"):
        st.code("\n".join(result.get('trace', [])) or "No trace", language="text")
    with st.expander("Audit log entry"):
        st.json(json.loads(json.dumps(result['log'], default=str)))


# ----------------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'question' not in st.session_state:
    st.session_state.question = ""

st.title("🏦 Credit Risk Query Engine")
st.caption(
    "Northbridge Bank · Commercial lending portfolio · Read-only. Ask a portfolio question in plain "
    "English. Every answer shows the SQL used, the raw data and a confidence score; queries that "
    "cannot be validated are escalated to an analyst."
)

with st.sidebar:
    st.header("Sample questions")

    def _use_sample():
        if st.session_state.sample_choice:
            st.session_state.question = st.session_state.sample_choice

    st.selectbox(
        "Pick one to fill the question box",
        options=[""] + load_sample_questions(),
        key="sample_choice",
        on_change=_use_sample,
    )

    st.header("Verified query library")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry['sql'], language="sql")

    with st.expander("Database schema"):
        st.code(database_schema, language="text")

    st.caption(f"Model: {MODEL_NAME} · Database: {DB_PATH} (read-only)")

with st.form("query_form"):
    st.text_area(
        "Your question",
        key="question",
        height=90,
        placeholder="e.g. What is the total outstanding by loan category?",
    )
    submitted = st.form_submit_button("Run query", type="primary")

if submitted:
    question = st.session_state.question.strip()
    if not question:
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Classifying, validating and running your query..."):
                result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
            st.session_state.last_result = result
            st.session_state.history.append(
                {'timestamp': datetime.now().isoformat(timespec='seconds'), **result['log']}
            )
        except Exception as e:
            st.session_state.last_result = None
            st.error(f"The pipeline failed before producing an answer: {e}")

if st.session_state.last_result is not None:
    render_result(st.session_state.last_result)

if st.session_state.history:
    st.divider()
    st.subheader("Audit trail (this session)")
    history_df = pd.DataFrame([
        {
            'Timestamp': h['timestamp'],
            'Question': h['user_question'],
            'Route': h['route'],
            'Query ID': h['query_id'],
            'Retry used': h['retry_used'],
            'Escalated': h['escalated'],
            'Confidence': format_confidence(h['confidence']),
            'Rows': h['row_count'],
        }
        for h in st.session_state.history
    ])
    st.dataframe(history_df)
    st.download_button(
        "Download full audit log (JSON)",
        json.dumps(st.session_state.history, indent=2, default=str).encode("utf-8"),
        file_name="audit_log.json",
        mime="application/json",
    )
