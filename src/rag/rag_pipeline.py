"""
Retrieval-augmented generation pipeline (Chapter 3.5 "RAG Augmented Response
System").

Retrieval strategy: the FULL workload profile (risk class + top influential
features + the student's free-text description, if given) is used as the
query — not just a topic keyword — per Chapter 3.3's "profile-based
retrieval" design intended to fix the panel's undifferentiated-response
concern. Retrieval runs separately against each of the three KB layers
(cultural / intervention / local) so every generated response draws on all
three, then the LLM synthesises them into one answer.

Two generation backends are supported — pick one via LLM_BACKEND below:
  - "ollama"    : free, fully local (matches your stated tech choice —
                  Llama 3 via Ollama). Requires `ollama serve` running and
                  `ollama pull llama3` done once. No API key needed.
  - "anthropic" : Claude via the Anthropic API. Requires ANTHROPIC_API_KEY
                  in your environment. Useful if you don't want to run a
                  local model on your laptop for the evaluation study.

Also exposes `generate_baseline(...)`, which answers the SAME prompt with NO
retrieved context — this is your baseline LLM for the RQ3/RQ4 comparison
study in Chapter 3.6.
"""
import os
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR  # noqa: E402
from rag.build_kb import EMBEDDING_MODEL, COLLECTION_NAME  # noqa: E402

LLM_BACKEND = os.environ.get("WORKLOADLENS_LLM_BACKEND", "ollama")  # "ollama" | "anthropic"
OLLAMA_MODEL = os.environ.get("WORKLOADLENS_OLLAMA_MODEL", "llama3")
ANTHROPIC_MODEL = os.environ.get("WORKLOADLENS_ANTHROPIC_MODEL", "claude-sonnet-4-6")

_embed_model = None
_collection = None


def _load():
    global _embed_model, _collection
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _embed_model, _collection


def build_query_text(risk_result: dict, free_text: str = "") -> str:
    """Combine the ML output + free-text description into one retrieval query."""
    parts = [f"Workload risk level: {risk_result.get('risk_class', 'Unknown')}."]
    features = risk_result.get("top_influential_features") or []
    if features:
        readable = ", ".join(f.replace("_", " ") for f in features)
        parts.append(f"Most influential workload factors for this student: {readable}.")
    if free_text:
        parts.append(f"Student's own description: {free_text}")
    return " ".join(parts)


def retrieve(query_text: str, k_per_layer: int = 2):
    _, collection = _load()
    model, _ = _load()
    query_embedding = model.encode([query_text]).tolist()

    retrieved = []
    for layer in ["cultural", "intervention", "local"]:
        res = collection.query(
            query_embeddings=query_embedding,
            n_results=k_per_layer,
            where={"layer": layer},
        )
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            retrieved.append({"text": doc, "metadata": meta, "distance": dist})
    return retrieved


def _format_context(chunks):
    lines = []
    for c in chunks:
        lines.append(f"[{c['metadata']['layer']}] {c['text']} (source: {c['metadata']['source']})")
    return "\n\n".join(lines)


SYSTEM_PROMPT = (
    "You are WorkloadLens, an academic time-management support assistant for "
    "Sri Lankan university students. Give practical, specific, non-clinical "
    "guidance grounded in the provided context. Never give medical or mental "
    "health advice — if the student's message suggests significant distress, "
    "tell them to contact their university's student welfare/counselling "
    "service instead of giving workload advice. Keep responses concise "
    "(150-250 words), concrete, and framed as options students and "
    "researchers in similar situations have found helpful, not as commands."
)


def _call_ollama(prompt: str) -> str:
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
         json={"model": OLLAMA_MODEL, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False, "keep_alive": "30m"},
        timeout=600,
    )
    if resp.status_code >= 400:
        print("OLLAMA ERROR BODY:", resp.text)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_llm(prompt: str) -> str:
    if LLM_BACKEND == "ollama":
        return _call_ollama(prompt)
    elif LLM_BACKEND == "anthropic":
        return _call_anthropic(prompt)
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")


def generate_rag_response(risk_result: dict, free_text: str = "") -> dict:
    query_text = build_query_text(risk_result, free_text)
    chunks = retrieve(query_text)
    context = _format_context(chunks)

    prompt = (
        f"Student workload profile:\n{query_text}\n\n"
        f"Relevant grounded knowledge (three layers: cultural, intervention, local):\n"
        f"{context}\n\n"
        f"Using the above context, write time-management guidance tailored "
        f"specifically to this student's workload profile."
    )
    answer = _call_llm(prompt)
    return {"answer": answer, "retrieved_chunks": chunks, "query_used": query_text}


def generate_baseline(risk_result: dict, free_text: str = "") -> dict:
    """Same prompt, NO retrieved context — the RQ3/RQ4 comparison baseline."""
    query_text = build_query_text(risk_result, free_text)
    prompt = (
        f"Student workload profile:\n{query_text}\n\n"
        f"Write time-management guidance tailored to this student's workload profile."
    )
    answer = _call_llm(prompt)
    return {"answer": answer, "query_used": query_text}


if __name__ == "__main__":
    example_risk = {
        "risk_class": "High",
        "top_influential_features": ["Q8 Overlap", "Q19 Sleep", "Q5 Part-time"],
    }
    result = generate_rag_response(
        example_risk,
        free_text="I have three assignments due the same week and a part-time job.",
    )
    print("RAG ANSWER:\n", result["answer"])
    print("\nRETRIEVED CHUNK IDS:", [c["metadata"]["file"] for c in result["retrieved_chunks"]])
