"""
Central configuration for WorkloadLens.
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "WorkloadLens_Real_Responses.xlsx"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
CHROMA_DIR = ROOT_DIR / "outputs" / "chroma_db"
KB_SOURCE_DIR = ROOT_DIR / "src" / "rag" / "sample_docs"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

for d in [DATA_PROCESSED, MODELS_DIR, CHROMA_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Raw survey columns -----------------------------------------------------

RAW_COLUMNS = [
    "Q1 Year", "Q2 Degree", "Q3 University", "Q4 Residence", "Q5 Part-time",
    "Q6 Assignments", "Q6 Exams", "Q6 Vivas", "Q6 Presentations", "Q7 Subjects",
    "Q8 Overlap", "Q9 Notice", "Q10 Study Time", "Q11 Group Work",
    "Q12 Assignment Difficulty", "Q12 Exam Difficulty", "Q12 Viva Difficulty",
    "Q13 New Subjects", "Q14 Untaught Content", "Q15 Self-study",
    "Q16 Self-study Hours", "Q17 Lost Time", "Q18 Study Space", "Q19 Sleep",
    "Q20 Energy", "Q20 Motivation", "Q20 Financial Pressure", "Q20 Family Pressure",
    "Q21 Late Tasks", "Q22 Missed Classes", "Q23 Time Management",
    "Q23 Work Quality", "Q24 Workload Impact", "Q25 AI Usefulness",
]

# Columns describing the population strata used for RQ1 sub-group analysis
STRATA_COLUMNS = ["Q1 Year", "Q2 Degree", "Q3 University"]

# Columns dropped from the ML feature matrix because they are *outcomes*
# (used only to build the target label), not predictors.
OUTCOME_COLUMNS = [
    "Q21 Late Tasks", "Q22 Missed Classes", "Q23 Time Management",
    "Q23 Work Quality", "Q24 Workload Impact",
]

# Q25 is dropped from modelling entirely (it's about AI-tool usefulness,
# not workload) but kept in the processed file for reference.
NON_FEATURE_COLUMNS = ["Q25 AI Usefulness"]

TARGET_COLUMN = "workload_risk"          # engineered 3-class label
RISK_CLASSES = ["Low", "Moderate", "High"]

RANDOM_STATE = 42
