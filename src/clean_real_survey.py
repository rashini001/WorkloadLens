
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import DATA_RAW, REPORTS_DIR, RAW_COLUMNS  # noqa: E402

# Position-based rename: the Nth non-Timestamp column in your Google Forms
# export -> the short internal name. If your form's question order ever
# changes, update this list to match (it must stay the same length/order as
# RAW_COLUMNS in config.py).
COLUMN_ORDER = RAW_COLUMNS  # 34 names, must match the export's column order

# Known answer-text inconsistencies (typos / inconsistent dash characters
# that crept in from form edits). Each entry: {column_name: {wrong: right}}.
# Extend this if preprocessing.py's encode_ordinals() still raises an
# "Unseen category" error after running this script — it will tell you
# exactly which column/value is still unmatched.
TEXT_CORRECTIONS = {
    "Q2 Degree": {
        "Art / Humanities": "Arts / Humanities",
    },
    "Q4 Residence": {
        "Rented accommoation": "Rented accommodation",
    },
    "Q7 Subjects": {
        "1 - 2 subjects": "1 – 2 subjects",     # hyphen -> en dash
        "3- 4 subjects": "3 – 4 subjects",       # missing space + hyphen -> en dash
    },
    "Q15 Self-study": {
        "Moderate – 3 to 5 extra hours": "Moderate – 3 – 5 extra hours",
    },
}

LIKERT_INT_COLUMNS = [
    "Q12 Assignment Difficulty", "Q12 Exam Difficulty", "Q12 Viva Difficulty",
    "Q20 Energy", "Q20 Motivation", "Q20 Financial Pressure", "Q20 Family Pressure",
    "Q23 Time Management", "Q23 Work Quality",
]


def clean(csv_path: Path):
    raw = pd.read_csv(csv_path)
    n_raw = len(raw)

    has_timestamp = "Timestamp" in raw.columns
    question_cols = [c for c in raw.columns if c != "Timestamp"]

    if len(question_cols) != len(COLUMN_ORDER):
        raise ValueError(
            f"Expected {len(COLUMN_ORDER)} question columns (after dropping Timestamp), "
            f"found {len(question_cols)}. Your export's column count doesn't match "
            f"config.RAW_COLUMNS — open this script and check COLUMN_ORDER / your CSV."
        )

    # IMPORTANT: check for accidental-resubmission duplicates on the FULL raw
    # row (Timestamp + every answer) BEFORE dropping Timestamp. With ~34
    # categorical questions that each only have 3-6 possible answers, it's
    # statistically expected that many DIFFERENT students will coincidentally
    # pick the exact same combination of answers — that is not a duplicate
    # submission. Only an identical Timestamp + identical answers together
    # means the same form submission was recorded twice.
    dup_mask = raw.duplicated(keep="first")
    n_duplicates = int(dup_mask.sum())
    raw = raw[~dup_mask]

    data = raw.drop(columns=["Timestamp"]) if has_timestamp else raw.copy()
    data.columns = COLUMN_ORDER

    # strip whitespace on every text cell, then apply known corrections
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].astype(str).str.strip().replace("nan", pd.NA)
    for col, mapping in TEXT_CORRECTIONS.items():
        if col in data.columns:
            data[col] = data[col].replace(mapping)

    # drop fully-blank rows, then any row with a remaining missing answer
    fully_blank = data[data.isna().all(axis=1)].index
    data = data.drop(index=fully_blank)
    incomplete = data[data.isna().any(axis=1)].index
    data = data.drop(index=incomplete)

    for col in LIKERT_INT_COLUMNS:
        data[col] = data[col].astype(float).astype(int)

    n_clean = len(data)

    log = {
        "N_raw_responses": n_raw,
        "dropped_fully_blank_rows": len(fully_blank),
        "dropped_incomplete_rows": len(incomplete),
        "dropped_exact_duplicate_rows": n_duplicates,
        "n_valid_responses": n_clean,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_cleaning_log.json", "w") as f:
        json.dump(log, f, indent=2)

    out_path = DATA_RAW.parent / "WorkloadLens_Real_Responses.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_excel(out_path, index=False)

    print(json.dumps(log, indent=2))
    print(f"\nCleaned file saved -> {out_path}")
    print(f"\nFor Chapter 5.4.1: '{log['N_raw_responses']} responses were received and "
          f"{log['n_valid_responses']} remained after data cleaning'")
    print("\nNext: open config.py and point DATA_RAW at this new file (or rename it to "
          "match DATA_RAW exactly), then run preprocessing.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to your Google Forms export CSV")
    args = parser.parse_args()
    clean(Path(args.csv_path))
