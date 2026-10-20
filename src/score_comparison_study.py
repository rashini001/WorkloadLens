"""
Scores the blind comparison study (Chapter 3.6 / 5.7) — matches your
Google Form's raw rater responses back to the private answer key (which
response was actually WorkloadLens vs baseline per scenario) and runs
paired statistical tests per criterion.

PREREQUISITES:
  1. outputs/reports/comparison_study_answer_key.json already exists
     (from generate_comparison_study.py)
  2. Your Google Form export has columns named like:
     "Rate RESPONSE A - Profile 3 [Relevance]"
     (the Google Forms grid format; Criterion is one of: Relevance, Actionability,
     Cultural fit, Specificity). The older "Scenario <N> - Response <A/B> - <Criterion>"
     format is also still accepted.
  3. pip install scipy

Usage:
    python score_comparison_study.py "path/to/your Google Form export.csv"
"""
import argparse
import json
import re

import pandas as pd
from scipy import stats

from config import REPORTS_DIR

# Google Forms grid export:  "Rate RESPONSE A - Profile 3 [Relevance]"
COLUMN_RE_GRID = re.compile(
    r"Rate\s+RESPONSE\s*([AB])\s*-\s*Profile\s*(\d+)\s*\[\s*(Relevance|Actionability|Cultural fit|Specificity)\s*\]",
    re.IGNORECASE,
)
# Older format:  "Scenario 3 - Response A - Relevance"
COLUMN_RE_OLD = re.compile(
    r"Scenario\s*(\d+)\s*-\s*Response\s*([AB])\s*-\s*(Relevance|Actionability|Cultural fit|Specificity)",
    re.IGNORECASE,
)
CRITERIA = ["Relevance", "Actionability", "Cultural fit", "Specificity"]


def parse_column(col):
    """Return (scenario_number, response_letter, criterion) or None."""
    m = COLUMN_RE_GRID.search(col)
    if m:
        letter, num, crit = m.group(1), m.group(2), m.group(3)
    else:
        m = COLUMN_RE_OLD.search(col)
        if not m:
            return None
        num, letter, crit = m.group(1), m.group(2), m.group(3)
    canonical = next(c for c in CRITERIA if c.lower() == crit.lower())
    return int(num), letter.upper(), canonical


def load_answer_key():
    path = REPORTS_DIR / "comparison_study_answer_key.json"
    with open(path) as f:
        key = json.load(f)
    # answer_key entries are in the SAME ORDER the scenarios were generated,
    # i.e. entry 0 = Scenario 1, entry 1 = Scenario 2, etc.
    return {i + 1: entry["a_is_workloadlens"] for i, entry in enumerate(key)}


def run(csv_path: str):
    df = pd.read_csv(csv_path)
    answer_key = load_answer_key()

    # long-format table: one row per (rater, scenario, criterion, system, score)
    records = []
    for col in df.columns:
        parsed = parse_column(col)
        if not parsed:
            continue
        scenario_num, response_letter, criterion = parsed
        if scenario_num not in answer_key:
            print(f"WARNING: column '{col}' references Scenario {scenario_num}, "
                  f"which isn't in the answer key (only {len(answer_key)} scenarios recorded) — skipping")
            continue
        a_is_workloadlens = answer_key[scenario_num]
        system = "WorkloadLens" if (response_letter == "A") == a_is_workloadlens else "Baseline"

        for rater_idx, score in pd.to_numeric(df[col], errors="coerce").items():
            if pd.isna(score):
                continue
            records.append({
                "rater": rater_idx, "scenario": scenario_num,
                "criterion": criterion, "system": system, "score": float(score),
            })

    if not records:
        raise ValueError(
            "No matching columns found. Check your Google Form question titles exactly "
            "look like 'Rate RESPONSE A - Profile 1 [Relevance]'. Columns found: "
            + ", ".join(df.columns[:6].astype(str)) + " ..."
        )

    long_df = pd.DataFrame(records)
    n_raters = long_df["rater"].nunique()
    n_scenarios = long_df["scenario"].nunique()

    results = {"n_raters": n_raters, "n_scenarios": n_scenarios, "per_criterion": {}}

    for criterion in CRITERIA:
        crit_df = long_df[long_df["criterion"] == criterion]
        # average each rater's scores across scenarios, per system, so each
        # rater contributes one paired observation (not one per scenario,
        # which would violate the paired-test independence assumption)
        pivot = crit_df.groupby(["rater", "system"])["score"].mean().unstack("system")
        pivot = pivot.dropna()  # keep only raters who rated both systems

        wl_scores = pivot["WorkloadLens"]
        baseline_scores = pivot["Baseline"]

        diff = wl_scores - baseline_scores
        sd_diff = diff.std()
        cohens_dz = round(float(diff.mean() / sd_diff), 3) if sd_diff > 0 else None
        shapiro_p = round(float(stats.shapiro(diff).pvalue), 4) if len(diff) >= 3 else None
        t_stat, t_p = stats.ttest_rel(wl_scores, baseline_scores)
        try:
            w_stat, w_p = stats.wilcoxon(wl_scores, baseline_scores)
        except ValueError:
            w_stat, w_p = None, None  # all differences are zero, or too few pairs

        results["per_criterion"][criterion] = {
            "n_paired_raters": len(pivot),
            "mean_workloadlens": round(float(wl_scores.mean()), 3),
            "sd_workloadlens": round(float(wl_scores.std()), 3),
            "mean_baseline": round(float(baseline_scores.mean()), 3),
            "sd_baseline": round(float(baseline_scores.std()), 3),
            "mean_difference": round(float(diff.mean()), 3),
            "cohens_dz": cohens_dz,
            "shapiro_p_of_differences": shapiro_p,
            "differences_look_normal": (shapiro_p >= 0.05) if shapiro_p is not None else None,
            "paired_t_test": {"t": round(float(t_stat), 3), "p": float(f"{t_p:.4g}")},
            "wilcoxon_signed_rank": (
                {"statistic": round(float(w_stat), 3), "p": float(f"{w_p:.4g}")}
                if w_stat is not None else None
            ),
        }

    out_path = REPORTS_DIR / "comparison_study_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")
    print(f"\n{n_raters} raters, {n_scenarios} scenarios scored.")
    print("\nFor Chapter 5.7, report mean_workloadlens vs mean_baseline per criterion, "
          "with the paired t-test (or Wilcoxon if your data isn't normally distributed) "
          "p-value as evidence of whether the difference is statistically significant (p < .05).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to your Google Form export CSV")
    args = parser.parse_args()
    run(args.csv_path)
