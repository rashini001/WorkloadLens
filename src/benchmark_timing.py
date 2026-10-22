
import argparse
import json
import platform
import statistics
import time

import predict
from rag import rag_pipeline
from config import REPORTS_DIR

EXAMPLE_PROFILE = {
    "Q1 Year": "Year 3", "Q2 Degree": "Computing / IT", "Q3 University": "NIBM",
    "Q4 Residence": "University hostel", "Q5 Part-time": "Yes - 10 to 20 hours per week",
    "Q6 Assignments": "3 or more", "Q7 Subjects": "5 – 6 subjects",
    "Q8 Overlap": "4 or more overlapping", "Q9 Notice": "1 – 3 days (very short notice)",
    "Q10 Study Time": "Required much more than available",
    "Q11 Group Work": "Yes – 1 group task",
    "Q12 Assignment Difficulty": 4, "Q12 Exam Difficulty": 3, "Q12 Viva Difficulty": 3,
    "Q13 New Subjects": "1 new or unfamiliar subject",
    "Q14 Untaught Content": "Some tasks required untaught content",
    "Q15 Self-study": "Moderate – 3 – 5 extra hours", "Q16 Self-study Hours": "3 – 4 hours",
    "Q17 Lost Time": "1 – 2 hours", "Q18 Study Space": "Sometimes available",
    "Q19 Sleep": "Less than 4 hours",
    "Q20 Energy": 2, "Q20 Motivation": 2, "Q20 Financial Pressure": 3, "Q20 Family Pressure": 3,
}
FREE_TEXT = "I have three assignments due the same week and a part-time job."


def hardware_description() -> str:
    """Best-effort machine description for the '[hardware description]'
    placeholder. Edit the returned string if you want to state exact CPU/RAM/
    GPU specs instead (more informative for a performance table)."""
    return f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}"


def time_stage(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def run(n_runs: int):
    stages = {"classification": [], "retrieval": [], "rag_generation": [], "baseline_generation": []}

    for i in range(n_runs):
        risk_result, t_classify = time_stage(predict.predict_risk, EXAMPLE_PROFILE)
        stages["classification"].append(t_classify)

        query_text = rag_pipeline.build_query_text(risk_result, FREE_TEXT)
        chunks, t_retrieve = time_stage(rag_pipeline.retrieve, query_text)
        stages["retrieval"].append(t_retrieve)

        _, t_rag = time_stage(rag_pipeline.generate_rag_response, risk_result, FREE_TEXT)
        stages["rag_generation"].append(t_rag)

        _, t_baseline = time_stage(rag_pipeline.generate_baseline, risk_result, FREE_TEXT)
        stages["baseline_generation"].append(t_baseline)

        print(f"run {i+1}/{n_runs} done")

    summary = {
        "n_runs": n_runs,
        "hardware_description": hardware_description(),
        "llm_backend": rag_pipeline.LLM_BACKEND,
        "stages": {
            name: {
                "mean_seconds": round(statistics.mean(times), 3),
                "max_seconds": round(max(times), 3),
                "min_seconds": round(min(times), 3),
            }
            for name, times in stages.items()
        },
    }
    summary["stages"]["total_rag_pipeline"] = {
        "mean_seconds": round(
            summary["stages"]["classification"]["mean_seconds"]
            + summary["stages"]["retrieval"]["mean_seconds"]
            + summary["stages"]["rag_generation"]["mean_seconds"], 3
        )
    }

    out_path = REPORTS_DIR / "performance_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (Table 5.4 source) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_path}")
    print(f"\nFor the Chapter 5.5 sentence: '{n_runs} runs on {summary['hardware_description']}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    run(args.runs)
