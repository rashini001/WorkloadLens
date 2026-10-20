"""
Scores the RAG system on faithfulness and answer relevance (RAGAS) across
your evaluation set (test_profiles.py) — fills in Chapter 5.6.2's
"[number] workload profiles", faithfulness score, relevance score.

PREREQUISITES before running:
  1. pip install ragas datasets langchain-ollama langchain-huggingface
  2. outputs/models/best_model.joblib must exist (train_models.py already run)
  3. kb_store built (workloadlens_kb.py build already run)
  4. Ollama running — strongly recommended: use the smaller llama3.2:3b
     model for THIS script specifically, not the full llama3. RAGAS's
     faithfulness/answer_relevancy metrics each make SEVERAL separate LLM
     calls per profile internally (breaking the answer into claims, then
     checking each one), so the grading overhead here is much higher than
     a single normal generation call — on CPU-only hardware, llama3 is
     usually too slow for this to finish in reasonable time.
        set WORKLOADLENS_OLLAMA_MODEL=llama3.2:3b
     (set this in the SAME terminal window, before running this script —
     environment variables don't carry over between terminal sessions)
  5. test_profiles.py filled in with your real evaluation set

FIX in this version: RAGAS defaults to running many jobs IN PARALLEL with a
short per-job timeout — fine against a cloud API, but wrong for a single
local Ollama instance, which can only actually process one request at a
time anyway. Running jobs in parallel against it just makes them all
compete for the same compute and time out together. This version uses
RAGAS's RunConfig to run jobs ONE AT A TIME with a much longer timeout,
which is slower per-job but far more likely to actually finish.

Usage:
    python evaluate_ragas.py
"""
import json

import predict
from rag import rag_pipeline
from config import REPORTS_DIR
from test_profiles import TEST_PROFILES


def run():
    try:
        from ragas import evaluate, RunConfig
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from datasets import Dataset
    except ImportError:
        raise SystemExit(
            "Run: pip install ragas datasets langchain-ollama langchain-huggingface"
        )

    if len(TEST_PROFILES) < 3:
        print(f"WARNING: only {len(TEST_PROFILES)} profiles in test_profiles.py — "
              f"fill in more (aim for 10-15) before running your real evaluation.\n")

    print(f"Using Ollama model: {rag_pipeline.OLLAMA_MODEL} "
          f"(set WORKLOADLENS_OLLAMA_MODEL=llama3.2:3b first if this is still 'llama3' "
          f"and you're on CPU-only hardware — see this file's docstring)\n")

    if rag_pipeline.LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        judge_llm = LangchainLLMWrapper(ChatOllama(model=rag_pipeline.OLLAMA_MODEL, temperature=0))
    elif rag_pipeline.LLM_BACKEND == "anthropic":
        from langchain_anthropic import ChatAnthropic
        judge_llm = LangchainLLMWrapper(ChatAnthropic(model=rag_pipeline.ANTHROPIC_MODEL, temperature=0))
    else:
        raise SystemExit(f"Unknown LLM_BACKEND: {rag_pipeline.LLM_BACKEND}")

    from langchain_huggingface import HuggingFaceEmbeddings
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    )

    # run jobs ONE AT A TIME (max_workers=1) with a generous timeout —
    # local CPU inference can genuinely take several minutes per call when
    # RAGAS is making multiple internal LLM calls per metric per profile
    run_config = RunConfig(timeout=900, max_workers=1, max_retries=2)

    rows = {"question": [], "answer": [], "contexts": []}
    for item in TEST_PROFILES:
        risk_result = predict.predict_risk(item["profile"])
        rag_result = rag_pipeline.generate_rag_response(
            risk_result, item["free_text"], profile=item["profile"]
        )
        rows["question"].append(rag_result["query_used"])
        rows["answer"].append(rag_result["answer"])
        rows["contexts"].append([c["text"] for c in rag_result["retrieved_chunks"]])
        print(f"done: {item['label']}")

    dataset = Dataset.from_dict(rows)
    print(f"\nGrading {len(TEST_PROFILES)} profiles one at a time — this can take a "
          f"while on local hardware (expect several minutes per profile). Do not "
          f"close this window even if it looks idle.\n")
    result = evaluate(
        dataset, metrics=[faithfulness, answer_relevancy],
        llm=judge_llm, embeddings=judge_embeddings,
        run_config=run_config,
    )
    df = result.to_pandas()

    n_timed_out = int(df["faithfulness"].isna().sum() + df["answer_relevancy"].isna().sum())
    if n_timed_out > 0:
        print(f"\nWARNING: {n_timed_out} score(s) are still missing/NaN even with the "
              f"longer timeout — your model may still be too slow for this hardware. "
              f"Consider switching WORKLOADLENS_OLLAMA_MODEL to an even smaller model, "
              f"or running this on fewer profiles at a time.")

    summary = {
        "n_profiles": len(TEST_PROFILES),
        "judge_backend": rag_pipeline.LLM_BACKEND,
        "judge_model": rag_pipeline.OLLAMA_MODEL if rag_pipeline.LLM_BACKEND == "ollama" else rag_pipeline.ANTHROPIC_MODEL,
        "n_timed_out_or_missing": n_timed_out,
        "mean_faithfulness": round(df["faithfulness"].mean(skipna=True), 3),
        "mean_answer_relevancy": round(df["answer_relevancy"].mean(skipna=True), 3),
        "per_profile": df[["faithfulness", "answer_relevancy"]].to_dict(orient="records"),
    }

    out_path = REPORTS_DIR / "ragas_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_path}")
    if n_timed_out == 0:
        print(f"\nFor Chapter 5.6.2: 'across {summary['n_profiles']} workload profiles... "
              f"mean faithfulness score was {summary['mean_faithfulness']} and the mean answer "
              f"relevance score was {summary['mean_answer_relevancy']}'")
    else:
        print(f"\nDo NOT cite these numbers yet — {n_timed_out} score(s) are still missing.")


if __name__ == "__main__":
    run()
