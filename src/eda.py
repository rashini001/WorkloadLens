
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_RAW, DATA_PROCESSED, REPORTS_DIR, TARGET_COLUMN
from preprocessing import load_raw, encode_ordinals, compute_risk_label


def run():
    raw = load_raw()
    encoded = encode_ordinals(raw)
    encoded[TARGET_COLUMN] = compute_risk_label(encoded)

    # Overall risk class distribution
    counts = encoded[TARGET_COLUMN].value_counts().reindex(["Low", "Moderate", "High"])
    ax = counts.plot(kind="bar", color=["#4CAF50", "#FFC107", "#F44336"])
    ax.set_title("Workload Risk Distribution (n=%d)" % len(encoded))
    ax.set_ylabel("Number of students")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "risk_distribution.png")
    plt.close()

    # Cross-tabs for RQ1 sub-group comparison
    for strata_col, filename in [
        ("Q1 Year", "risk_by_year.png"),
        ("Q2 Degree", "risk_by_degree.png"),
        ("Q3 University", "risk_by_university.png"),
    ]:
        # re-map Q1 Year back to labels for a readable chart
        cross = pd.crosstab(raw[strata_col], encoded[TARGET_COLUMN], normalize="index")
        cross = cross.reindex(columns=["Low", "Moderate", "High"])
        ax = cross.plot(kind="bar", stacked=True, color=["#4CAF50", "#FFC107", "#F44336"])
        ax.set_title(f"Workload Risk by {strata_col}")
        ax.set_ylabel("Proportion")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / filename)
        plt.close()

    # Summary stats table for the dissertation appendix
    summary = raw.describe(include="all").transpose()
    summary.to_csv(REPORTS_DIR / "summary_stats.csv")

    print(f"EDA charts + summary_stats.csv written to {REPORTS_DIR}")


if __name__ == "__main__":
    run()
