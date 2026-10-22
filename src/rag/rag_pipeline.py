"""
Retrieval-augmented generation pipeline (Chapter 3.5).

UPDATED to use workloadlens_kb.py's search() as the retrieval backend
instead of the earlier markdown-chunk ChromaDB collection — this is the
real, PDF-sourced, 18+ document knowledge base, with retrieval filtered by
the student's institution type and discipline 

Generation backend unchanged: set WORKLOADLENS_LLM_BACKEND=ollama (default)
or =anthropic, same as before.
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # project root, where workloadlens_kb.py lives
import workloadlens_kb  # noqa: E402

LLM_BACKEND = os.environ.get("WORKLOADLENS_LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.environ.get("WORKLOADLENS_OLLAMA_MODEL", "llama3")
ANTHROPIC_MODEL = os.environ.get("WORKLOADLENS_ANTHROPIC_MODEL", "claude-sonnet-4-6")

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


# ---- maps a student's survey answers to workloadlens_kb's filter labels ----
def infer_institution(profile: dict) -> str:
    """Maps Q3 University's answer to workloadlens_kb's institution labels
    ("state" / "non_state"). Extend this if you add more University options."""
    university = (profile.get("Q3 University") or "").lower()
    if "government" in university:
        return "state"
    if "nibm" in university or "private" in university:
        return "non_state"
    return None  # unknown -> no institution filter applied


def infer_discipline(profile: dict) -> str:
    """Maps Q2 Degree's answer to workloadlens_kb's discipline labels. Your
    survey currently has no medical/allied-health degree option, so those KB
    documents will only surface via workloadlens_kb's own relax-filter
    fallback, not a direct match — that's expected, not a bug."""
    degree = (profile.get("Q2 Degree") or "").lower()
    if "business" in degree or "management" in degree:
        return "management"
    return None  # general / unmapped -> no discipline filter applied


def build_query_text(risk_result: dict, free_text: str = "") -> str:
    parts = [f"Workload risk level: {risk_result.get('risk_class', 'Unknown')}."]
    features = risk_result.get("top_influential_features") or []
    if features:
        readable = ", ".join(f.replace("_", " ") for f in features)
        parts.append(f"Most influential workload factors for this student: {readable}.")
    if free_text:
        parts.append(f"Student's own description: {free_text}")
    return " ".join(parts)


def retrieve(query_text: str, profile: dict = None, k_per_layer: int = 2):
    institution = infer_institution(profile) if profile else None
    discipline = infer_discipline(profile) if profile else None
    results_by_layer = workloadlens_kb.search(
        query_text, institution=institution, discipline=discipline, k=k_per_layer
    )
    # flatten workloadlens_kb's {layer_name: [(id, text, meta, dist), ...]}
    # into the flat chunk-list shape the rest of this file expects
    flat = []
    for layer_name, hits in results_by_layer.items():
        for cid, text, meta, dist in hits:
            flat.append({"text": text, "metadata": meta, "distance": dist, "layer_name": layer_name})
    return flat


def _format_context(chunks):
    lines = []
    for c in chunks:
        meta = c["metadata"]
        citation = f"{meta.get('title', 'Unknown source')} ({meta.get('year', 'n.d.')})"
        lines.append(f"[{c['layer_name']}] {c['text']} (source: {citation})")
    return "\n\n".join(lines)


def _call_ollama(prompt: str) -> str:
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=500, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_llm(prompt: str) -> str:
    if LLM_BACKEND == "ollama":
        return _call_ollama(prompt)
    elif LLM_BACKEND == "anthropic":
        return _call_anthropic(prompt)
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")


def generate_rag_response(risk_result: dict, free_text: str = "", profile: dict = None) -> dict:
    query_text = build_query_text(risk_result, free_text)
    chunks = retrieve(query_text, profile=profile)
    context = _format_context(chunks)

    prompt = (
        f"Student workload profile:\n{query_text}\n\n"
        f"Relevant grounded knowledge (three layers: Sri Lankan context, interventions, local resources):\n"
        f"{context}\n\n"
        f"Using the above context, write time-management guidance tailored "
        f"specifically to this student's workload profile."
    )
    answer = _call_llm(prompt)
    return {"answer": answer, "retrieved_chunks": chunks, "query_used": query_text}


def generate_baseline(risk_result: dict, free_text: str = "") -> dict:
    query_text = build_query_text(risk_result, free_text)
    prompt = (
        f"Student workload profile:\n{query_text}\n\n"
        f"Write time-management guidance tailored to this student's workload profile."
    )
    answer = _call_llm(prompt)
    return {"answer": answer, "query_used": query_text}


if __name__ == "__main__":
    example_risk = {"risk_class": "High", "top_influential_features": ["Q8 Overlap", "Q19 Sleep"]}
    example_profile = {"Q2 Degree": "Business / Management", "Q3 University": "NIBM"}
    result = generate_rag_response(
        example_risk, free_text="I have three assignments due the same week and a part-time job.",
        profile=example_profile,
    )
    print("RAG ANSWER:\n", result["answer"])
    print("\nRETRIEVED:", [(c["layer_name"], c["metadata"].get("title")) for c in result["retrieved_chunks"]])
