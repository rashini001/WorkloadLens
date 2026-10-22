
import json

import pandas as pd

from config import REPORTS_DIR
from preprocessing import load_raw, encode_ordinals, compute_risk_label
from config import TARGET_COLUMN


def crosstab_table(raw_col: pd.Series, risk: pd.Series, label: str):
    counts = pd.crosstab(raw_col, risk)
    counts = counts.reindex(columns=["Low", "Moderate", "High"], fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    totals = counts.sum(axis=1)

    print(f"\n=== {label} ===")
    rows = []
    for category in counts.index:
        row = {
            "category": category,
            "n": int(totals[category]),
            "low_pct": round(float(pct.loc[category, "Low"]), 1),
            "moderate_pct": round(float(pct.loc[category, "Moderate"]), 1),
            "high_pct": round(float(pct.loc[category, "High"]), 1),
        }
        rows.append(row)
        print(f"  {category}: n={row['n']}  Low={row['low_pct']}%  "
              f"Moderate={row['moderate_pct']}%  High={row['high_pct']}%")
    return rows


def run():
    raw = load_raw()
    encoded = encode_ordinals(raw)
    risk = compute_risk_label(encoded)

    overall_counts = risk.value_counts()
    overall_pct = (overall_counts / overall_counts.sum() * 100).round(1)
    print("=== Overall ===")
    for cls in ["Low", "Moderate", "High"]:
        print(f"  {cls}: n={overall_counts.get(cls, 0)}  {overall_pct.get(cls, 0.0)}%")

    results = {
        "overall": {cls: {"n": int(overall_counts.get(cls, 0)), "pct": float(overall_pct.get(cls, 0.0))}
                    for cls in ["Low", "Moderate", "High"]},
        "by_year": crosstab_table(raw["Q1 Year"], risk, "By year of study"),
        "by_degree": crosstab_table(raw["Q2 Degree"], risk, "By degree"),
        "by_institution": crosstab_table(raw["Q3 University"], risk, "By institution"),
    }

    out_path = REPORTS_DIR / "exact_risk_crosstabs.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run()
