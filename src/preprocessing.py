
import json
import pandas as pd
import numpy as np

from config import (
    DATA_RAW, DATA_PROCESSED, RAW_COLUMNS, OUTCOME_COLUMNS,
    NON_FEATURE_COLUMNS, TARGET_COLUMN, RISK_CLASSES,
)

# ---------------------------------------------------------------------------
# Ordinal encodings: every survey answer that has a natural "more/less" order
# is mapped to an integer so the ML models can use it as a continuous-ish
# feature. Nominal (no order) columns are left as strings and one-hot encoded
# later by pandas.get_dummies in build_feature_matrix().
# ---------------------------------------------------------------------------
ORDINAL_MAPS = {
    "Q1 Year": {"Year 1": 1, "Year 2": 2, "Year 3": 3, "Year 4 / Final Year": 4},
    "Q5 Part-time": {
        "No": 0,
        "Yes - less than 10 hours per week": 1,
        "Yes - 10 to 20 hours per week": 2,
        "Yes - more than 20 hours per week": 3,
    },
    "Q6 Assignments": {"0": 0, "1": 1, "2": 2, "3 or more": 3},
    "Q6 Exams": {"0": 0, "1": 1, "2": 2, "3 or more": 3},
    "Q6 Vivas": {"0": 0, "1": 1, "2": 2, "3 or more": 3},
    "Q6 Presentations": {"0": 0, "1": 1, "2": 2, "3 or more": 3},
    "Q7 Subjects": {
        "1 – 2 subjects": 1, "3 – 4 subjects": 2,
        "5 – 6 subjects": 3, "7 or more subjects": 4,
    },
    "Q8 Overlap": {
        "None – all spread out": 0, "2 overlapping": 2,
        "3 overlapping": 3, "4 or more overlapping": 4,
    },
    "Q9 Notice": {
        "More than 2 weeks": 0, "1 – 2 weeks": 1,
        "4 – 7 days": 2, "1 – 3 days (very short notice)": 3,
    },
    "Q10 Study Time": {
        "Had more time than required": 0, "Roughly matched": 1,
        "Required slightly more than available": 2,
        "Required much more than available": 3,
    },
    "Q11 Group Work": {
        "No group work": 0, "Yes – 1 group task": 1, "Yes – 2 or more group tasks": 2,
    },
    "Q13 New Subjects": {
        "None – all familiar subjects": 0, "1 new or unfamiliar subject": 1,
        "2 new or unfamiliar subjects": 2, "3 or more new or unfamiliar subjects": 3,
    },
    "Q14 Untaught Content": {
        "No – all tasks matched taught content": 0,
        "Some tasks required untaught content": 1,
        "Most tasks required untaught content": 2,
    },
    "Q15 Self-study": {
        "None – lecture notes were fully sufficient": 0,
        "A little – 1 to 2 extra hours": 1,
        "Moderate – 3 – 5 extra hours": 2,
        "A lot – more than 5 extra hours": 3,
    },
    "Q16 Self-study Hours": {
        "Less than 1 hour": 0, "1 – 2 hours": 1, "3 – 4 hours": 2, "5 or more hours": 3,
    },
    "Q17 Lost Time": {
        "Less than 1 hour": 0, "1 – 2 hours": 1, "3 – 4 hours": 2, "More than 4 hours": 3,
    },
    "Q18 Study Space": {
        "Always available": 0, "Sometimes available": 1, "Rarely or never available": 2,
    },
    "Q19 Sleep": {
        "Less than 4 hours": 0, "4 – 5 hours": 1, "6 – 7 hours": 2, "8 hours or more": 3,
    },
    # --- outcome questions (used for the target label, not as features) ---
    "Q21 Late Tasks": {
        "No – all submitted on time": 0, "1 task submitted late": 1,
        "2 tasks submitted late": 2, "3 or more tasks late or missed completely": 3,
    },
    "Q22 Missed Classes": {
        "No – attended all classes": 0, "Missed 1 class": 1,
        "Missed 2 – 3 classes": 2, "Missed 4 or more classes": 3,
    },
    "Q24 Workload Impact": {
        "No negative impact – performed as expected": 0,
        "Slight negative impact": 1,
        "Moderate negative impact": 2,
        "Significant negative impact – much worse than usual": 3,
    },
    "Q25 AI Usefulness": {
        "Not useful at all": 0, "Slightly useful": 1, "Moderately useful": 2,
        "Very useful": 3, "Extremely useful – I would use this regularly": 4,
    },
}

# Q12/Q20/Q23 items are already 1–5 Likert integers in the raw file — no
# mapping needed, they're used as-is.

# Nominal columns (no inherent order) -> one-hot encoded
NOMINAL_COLUMNS = ["Q2 Degree", "Q3 University", "Q4 Residence"]


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(DATA_RAW)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")
    return df[RAW_COLUMNS].copy()


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        unseen = set(df[col].dropna().unique()) - set(mapping.keys())
        if unseen:
            raise ValueError(f"Unseen category in '{col}': {unseen}. Update ORDINAL_MAPS.")
        df[col] = df[col].map(mapping)
    return df


def compute_risk_label(df: pd.DataFrame) -> pd.Series:
    """
    Combine the four outcome questions into one 0–12 composite risk score,
    then bucket it into Low / Moderate / High using tertile-style fixed cut
    points. Q23 Time Management is a self-rated 1 (poor) – 5 (excellent)
    scale, so it is inverted (5 - rating) before summing so that every
    component of the score points the same direction (higher = worse).
    """
    late = df["Q21 Late Tasks"]                 # 0-3, higher = worse
    missed = df["Q22 Missed Classes"]            # 0-3, higher = worse
    impact = df["Q24 Workload Impact"]           # 0-3, higher = worse
    time_mgmt_inverted = 5 - df["Q23 Time Management"]  # 0-4, higher = worse

    score = late + missed + impact + time_mgmt_inverted  # range 0-13

    def bucket(s):
        if s <= 4:
            return "Low"
        elif s <= 8:
            return "Moderate"
        else:
            return "High"

    return score.apply(bucket)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal columns, drop outcome/non-feature columns."""
    features = df.drop(columns=OUTCOME_COLUMNS + NON_FEATURE_COLUMNS)
    features = pd.get_dummies(features, columns=NOMINAL_COLUMNS, prefix=NOMINAL_COLUMNS)
    return features


def run():
    raw = load_raw()
    encoded = encode_ordinals(raw)
    encoded[TARGET_COLUMN] = compute_risk_label(encoded)

    features = build_feature_matrix(encoded)
    features[TARGET_COLUMN] = encoded[TARGET_COLUMN]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "processed.csv"
    features.to_csv(out_path, index=False)

    # Save the ordinal maps + final feature column order so predict.py can
    # encode a single new student profile identically at inference time.
    meta = {
        "ordinal_maps": ORDINAL_MAPS,
        "nominal_columns": NOMINAL_COLUMNS,
        "feature_columns": [c for c in features.columns if c != TARGET_COLUMN],
        "risk_classes": RISK_CLASSES,
    }
    with open(DATA_PROCESSED / "preprocessing_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(features)} rows x {len(features.columns)} columns -> {out_path}")
    print(encoded[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    run()
