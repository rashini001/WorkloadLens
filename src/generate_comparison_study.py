
import csv
import json
import random

import predict
from rag import rag_pipeline
from config import REPORTS_DIR
from test_profiles import TEST_PROFILES


def run():
    if len(TEST_PROFILES) < 3:
        print(f"WARNING: only {len(TEST_PROFILES)} profiles in test_profiles.py — "
              f"fill in more (aim for 10-15) before running your real study.\n")

    rater_rows = []
    answer_key = []

    for item in TEST_PROFILES:
        risk_result = predict.predict_risk(item["profile"])
        rag_result = rag_pipeline.generate_rag_response(
            risk_result, item["free_text"], profile=item["profile"]
        )
        baseline_result = rag_pipeline.generate_baseline(risk_result, item["free_text"])

        if random.random() < 0.5:
            response_a, response_b = rag_result["answer"], baseline_result["answer"]
            a_is_workloadlens = True
        else:
            response_a, response_b = baseline_result["answer"], rag_result["answer"]
            a_is_workloadlens = False

        rater_rows.append({
            "profile_label": item["label"],
            "student_situation": item["free_text"],
            "response_a": response_a,
            "response_b": response_b,
        })
        answer_key.append({
            "profile_label": item["label"],
            "a_is_workloadlens": a_is_workloadlens,
        })
        print(f"done: {item['label']}")

    rater_path = REPORTS_DIR / "comparison_study_for_raters.csv"
    with open(rater_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["profile_label", "student_situation", "response_a", "response_b"])
        writer.writeheader()
        writer.writerows(rater_rows)

    key_path = REPORTS_DIR / "comparison_study_answer_key.json"
    with open(key_path, "w") as f:
        json.dump(answer_key, f, indent=2)

    print(f"\nRater-facing CSV (build your Google Form from this) -> {rater_path}")
    print(f"Answer key (keep PRIVATE until scoring) -> {key_path}")
    print(f"\n{len(TEST_PROFILES)} profiles generated.")
    print("\nNext: build a Google Form from the CSV above, send it to your raters, "
          "collect responses, THEN come back for the scoring step.")


if __name__ == "__main__":
    run()
