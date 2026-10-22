
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
