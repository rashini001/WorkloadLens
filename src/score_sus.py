
import argparse
import json

import pandas as pd

from config import REPORTS_DIR

SUS_QUESTIONS = [
    "I think that I would like to use this system frequently",
    "I found the system unnecessarily complex",
    "I thought the system was easy to use",
    "I think that I would need the support of a technical person to use this system",
    "I found the various functions in this system were well integrated",
    "I thought there was too much inconsistency in this system",
    "I would imagine that most people would learn to use this system very quickly",
    "I found the system very cumbersome to use",
    "I felt very confident using the system",
    "I needed to learn a lot of things before I could get going with this system",
]


def score_row(ratings: list) -> float:
    if len(ratings) != 10:
        raise ValueError(f"Expected 10 ratings, got {len(ratings)}")
    total = 0
    for i, r in enumerate(ratings):
        item_num = i + 1
        total += (r - 1) if item_num % 2 == 1 else (5 - r)
    return total * 2.5


def run(csv_path: str, skip_columns: int):
    df = pd.read_csv(csv_path)
    rating_cols = df.columns[skip_columns:skip_columns + 10]
    if len(rating_cols) != 10:
        raise ValueError(
            f"Found {len(rating_cols)} rating columns after skipping {skip_columns} "
            f"— expected 10. Check --skip-columns matches your export's layout."
        )

    scores = df[rating_cols].apply(lambda row: score_row(list(row)), axis=1)

    result = {
        "n_respondents": len(df),
        "mean_sus_score": round(float(scores.mean()), 2),
        "sd_sus_score": round(float(scores.std()), 2),
        "min_score": round(float(scores.min()), 2),
        "max_score": round(float(scores.max()), 2),
        "above_68_benchmark": bool(scores.mean() >= 68),
    }

    out_path = REPORTS_DIR / "sus_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")
    print(f"\nFor Chapter 5.8: 'mean SUS score of {result['mean_sus_score']} (SD = {result['sd_sus_score']}), "
          f"n = {result['n_respondents']}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", help="Path to your SUS Google Form export CSV")
    parser.add_argument("--skip-columns", type=int, default=1,
                         help="Number of non-rating columns before the 10 SUS ratings start "
                              "(default 1, for a Timestamp column)")
    parser.add_argument("--show-questions", action="store_true",
                         help="Print the 10 standard SUS questions (for building your Google Form) and exit")
    args = parser.parse_args()

    if args.show_questions or not args.csv_path:
        print("Standard SUS questions, in order (all rated 1=Strongly disagree to 5=Strongly agree):\n")
        for i, q in enumerate(SUS_QUESTIONS, 1):
            print(f"{i}. {q}")
        if not args.csv_path:
            raise SystemExit("\nPass a CSV path to actually score responses.")
    else:
        run(args.csv_path, args.skip_columns)
