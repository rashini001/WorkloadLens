"""
Computes Cronbach's alpha for the survey's Likert-scale items — fills
Chapter 5.4.1's "Cronbach's alpha of [α], which is [above / below] 0.70".

    python cronbach_alpha.py

IMPORTANT: run this on your REAL survey data (after preprocessing.py has
processed your real export, not the synthetic file), since alpha on
synthetic/random data is meaningless.

LIKERT_ITEMS below defaults to every 1-5 rated item in the survey (Q12 task-
difficulty items + Q20 wellbeing/pressure items + Q23 self-rated outcomes).
Reliability is normally reported per *construct* (a set of items meant to
measure the same underlying thing), not across unrelated constructs lumped
together — check with your supervisor whether Q12/Q20/Q23 are meant to form
one scale or separate sub-scales, and adjust LIKERT_ITEMS (or compute alpha
per group) accordingly before you quote a single alpha value in Chapter 5.
"""
import json
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, REPORTS_DIR

LIKERT_ITEMS = [
    "Q12 Assignment Difficulty", "Q12 Exam Difficulty", "Q12 Viva Difficulty",
    "Q20 Energy", "Q20 Motivation", "Q20 Financial Pressure", "Q20 Family Pressure",
    "Q23 Time Management", "Q23 Work Quality",
]


def cronbach_alpha(df: pd.DataFrame) -> float:
    items = df.dropna()
    k = items.shape[1]
    item_variances = items.var(axis=0, ddof=1)
    total_variance = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_variances.sum() / total_variance)


def run():
    df = pd.read_csv(DATA_PROCESSED / "processed.csv")
    missing = [c for c in LIKERT_ITEMS if c not in df.columns]
    if missing:
        raise ValueError(f"These Likert columns aren't in processed.csv: {missing}")

    alpha = cronbach_alpha(df[LIKERT_ITEMS])
    verdict = "above" if alpha >= 0.70 else "below"

    result = {"items_used": LIKERT_ITEMS, "n_respondents": len(df), "alpha": round(float(alpha), 3), "verdict": verdict}
    out_path = REPORTS_DIR / "cronbach_alpha.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nFor Chapter 5.4.1: 'Cronbach's alpha of {result['alpha']}, which is {verdict} the 0.70 threshold'")


if __name__ == "__main__":
    run()
