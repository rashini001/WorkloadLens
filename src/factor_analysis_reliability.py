"""
Exploratory Factor Analysis (EFA)-based reliability check — the correct
single-wave, purely data-driven alternative to Cronbach's alpha when your
items are a priori independent indicators rather than a known multi-item
scale (see the discussion that led here: alpha on Q12/Q20/Q23 gave ~0.01-0.14,
which is expected and NOT a data-quality problem for independent indicators,
but isn't a reportable "reliability" number on its own).

What this does, in order (this is the standard sequence reported in methods
sections, so Chapter 5.4.1 can cite each step):

1. KMO (Kaiser-Meyer-Olkin) test — checks whether your items have enough
   shared correlation to make factor analysis meaningful at all.
     KMO >= 0.60 is usually considered the minimum acceptable; below that,
     factor analysis isn't appropriate for this item set, which is itself a
     valid, citable, quantitative finding (not a failure).
2. Bartlett's test of sphericity — checks whether the items correlate with
   each other at all (p < .05 means yes, correlations are not just noise).
3. If both checks pass: runs EFA (principal axis factoring), extracts
   factors with eigenvalue > 1, and shows which items load on which factor.
4. For any factor with 2+ items, computes Cronbach's alpha WITHIN that
   factor — this is the methodologically correct grouping to report alpha
   on, because it's derived from the data rather than guessed from the
   questionnaire's question numbering.
5. If KMO/Bartlett's fail, or no coherent factor structure emerges: this
   prints the exact numbers to cite as evidence that these items are
   independent single-item indicators (not a scale), which is the
   Path-B-with-real-statistics-behind-it version of that argument.

    pip install factor_analyzer
    python factor_analysis_reliability.py

Run on your REAL cleaned survey data (DATA_RAW), same as cronbach_alpha.py.
"""
import json
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

from config import DATA_RAW, REPORTS_DIR

# The 9 subjective 1-5 self-rating items — the natural candidate set for
# factor analysis (objective count/ordinal items like Q8 Overlap or Q9
# Notice are a different kind of measure and aren't normally mixed into a
# psychological-construct factor analysis). Extend this list if your
# supervisor wants a broader item set tested.
CANDIDATE_ITEMS = [
    "Q12 Assignment Difficulty", "Q12 Exam Difficulty", "Q12 Viva Difficulty",
    "Q20 Energy", "Q20 Motivation", "Q20 Financial Pressure", "Q20 Family Pressure",
    "Q23 Time Management", "Q23 Work Quality",
]


def cronbach_alpha(df: pd.DataFrame) -> float:
    items = df.dropna()
    k = items.shape[1]
    if k < 2:
        return None
    item_variances = items.var(axis=0, ddof=1)
    total_variance = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - item_variances.sum() / total_variance)


def run():
    df = pd.read_excel(DATA_RAW)
    data = df[CANDIDATE_ITEMS].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(data)

    kmo_per_item, kmo_overall = calculate_kmo(data)
    chi_square, p_value = calculate_bartlett_sphericity(data)

    # per-item KMO, weakest first — shows which items (if any) are dragging
    # the overall score down, so you can see whether dropping 1-2 items
    # would plausibly cross the 0.60 threshold before you try it
    per_item_kmo = sorted(
        [{"item": item, "kmo": round(float(k), 3)} for item, k in zip(CANDIDATE_ITEMS, kmo_per_item)],
        key=lambda x: x["kmo"],
    )

    result = {
        "n_respondents": n,
        "items_tested": CANDIDATE_ITEMS,
        "kmo_overall": round(float(kmo_overall), 3),
        "kmo_per_item_weakest_first": per_item_kmo,
        "bartlett_chi_square": round(float(chi_square), 2),
        "bartlett_p_value": round(float(p_value), 4),
        "factor_analysis_appropriate": bool(kmo_overall >= 0.60 and p_value < 0.05),
    }

    if result["factor_analysis_appropriate"]:
        fa = FactorAnalyzer(rotation="varimax")
        fa.fit(data)
        ev, _ = fa.get_eigenvalues()
        n_factors = int((ev > 1).sum())

        if n_factors >= 1:
            fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
            fa.fit(data)
            loadings = pd.DataFrame(fa.loadings_, index=CANDIDATE_ITEMS,
                                     columns=[f"Factor{i+1}" for i in range(n_factors)])

            # assign each item to the factor it loads on most strongly (|loading| >= 0.40)
            factor_groups = {f"Factor{i+1}": [] for i in range(n_factors)}
            for item in CANDIDATE_ITEMS:
                row = loadings.loc[item]
                best = row.abs().idxmax()
                if abs(row[best]) >= 0.40:
                    factor_groups[best].append(item)

            result["n_factors_extracted"] = n_factors
            result["factor_loadings"] = loadings.round(3).to_dict()
            result["factor_groups"] = {k: v for k, v in factor_groups.items() if v}
            result["alpha_per_factor"] = {
                name: round(cronbach_alpha(data[items]), 3) if len(items) >= 2 else None
                for name, items in factor_groups.items() if items
            }
        else:
            result["n_factors_extracted"] = 0
            result["note"] = "No factors with eigenvalue > 1 — no coherent underlying structure found."
    else:
        result["note"] = (
            "KMO and/or Bartlett's test indicate these items do NOT share enough "
            "common variance for factor analysis to be meaningful — this is itself "
            "quantitative evidence that these are independent single-item indicators "
            "rather than a multidimensional scale, not a data-quality failure."
        )

        # sensitivity check: does dropping the 1-2 weakest items push KMO over 0.60?
        sensitivity = []
        remaining = list(CANDIDATE_ITEMS)
        for _ in range(min(3, len(CANDIDATE_ITEMS) - 4)):  # never drop below 4 items
            weakest = min(remaining, key=lambda it: next(
                p["kmo"] for p in per_item_kmo if p["item"] == it
            ))
            remaining = [it for it in remaining if it != weakest]
            sub_data = data[remaining]
            _, sub_kmo = calculate_kmo(sub_data)
            sensitivity.append({
                "dropped": weakest, "items_remaining": len(remaining),
                "kmo_after_dropping": round(float(sub_kmo), 3),
            })
            if sub_kmo >= 0.60:
                break
        result["sensitivity_check_dropping_weakest_items"] = sensitivity
        result["sensitivity_check_reached_060"] = any(s["kmo_after_dropping"] >= 0.60 for s in sensitivity)

    out_path = REPORTS_DIR / "factor_analysis_reliability.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps(result, indent=2, default=str))
    print(f"\nSaved -> {out_path}")

    if result["factor_analysis_appropriate"] and result.get("n_factors_extracted"):
        p_str = "< .05" if result["bartlett_p_value"] < .05 else f"= {result['bartlett_p_value']}"
        print(f"\nFor Chapter 5.4.1: 'KMO = {result['kmo_overall']}, Bartlett's test "
              f"chi-square({len(CANDIDATE_ITEMS)}) = {result['bartlett_chi_square']}, "
              f"p {p_str}, indicating factor analysis was appropriate. "
              f"{result['n_factors_extracted']} factor(s) were extracted; "
              f"reliability (Cronbach's alpha) for each empirically-derived factor is reported in Table X.'")
    else:
        print(f"\nFor Chapter 5.4.1: 'KMO = {result['kmo_overall']}, Bartlett's test "
              f"p = {result['bartlett_p_value']}, indicating the items did not share "
              f"sufficient common variance for factor analysis, consistent with their "
              f"design as independent single-item workload indicators rather than a "
              f"unidimensional scale.'")


if __name__ == "__main__":
    run()
