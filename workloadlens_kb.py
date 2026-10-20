"""
WorkloadLens knowledge base: build + search

Setup (once):
    pip install pandas openpyxl chromadb sentence-transformers pypdf beautifulsoup4 lxml

Folder layout:
    workloadlens_kb.py
    WorkloadLens_KB_metadata.xlsx
    docs/            <- put each document here, named by its ID: 1.pdf, 12.pdf, 26.html, 29.txt ...

Build the knowledge base:
    python workloadlens_kb.py build

Ask a question (profile options are optional):
    python workloadlens_kb.py search "I have 4 deadlines this week and I keep putting things off" \
        --institution non_state --discipline computing
"""

import argparse
import re
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

# ---------------- settings ----------------
# Paths are resolved relative to THIS FILE's location, not the current
# working directory — important because this module gets imported from
# rag/rag_pipeline.py, which runs from inside api/ (a different folder) when
# the FastAPI app is running. Without this, paths that "worked" when you
# ran `python workloadlens_kb.py build` by hand would silently break (or
# build a second, empty database) once imported by the API.
_BASE_DIR = Path(__file__).resolve().parent
META_FILE = _BASE_DIR / "WorkloadLens_KB_metadata.xlsx"
DOCS_DIR = _BASE_DIR / "docs"
DB_DIR = str(_BASE_DIR / "kb_store")
COLLECTION = "workloadlens"

# all-mpnet-base-v2 reads up to 384 tokens, so chunks of ~300 fit without being cut off.
# (all-MiniLM-L6-v2 stops at 256 tokens, so it would cut your 200-400 token chunks.)
MODEL_NAME = "all-mpnet-base-v2"
CHUNK_TOKENS = 300
OVERLAP_TOKENS = 40
MIN_CHUNK_TOKENS = 40  # throw away tiny leftovers

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
_tokenizer = None


def tokenizer():
    """Load the same tokenizer the embedding model uses, so token counts are accurate."""
    global _tokenizer
    if _tokenizer is None:
        from sentence_transformers import SentenceTransformer
        _tokenizer = SentenceTransformer(MODEL_NAME).tokenizer
    return _tokenizer


def n_tokens(text):
    return len(tokenizer().encode(text, add_special_tokens=False))


# ---------------- reading documents ----------------
def read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_html(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text("\n")


def read_doc(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".html", ".htm"):
        return read_html(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)   # join words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # cut the reference list so citations don't fill the knowledge base
    matches = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", text, flags=re.I))
    if matches and matches[-1].start() > len(text) * 0.6:
        text = text[: matches[-1].start()]
    return text.strip()


# ---------------- chunking ----------------
def split_sentences(text):
    text = re.sub(r"\s*\n\s*", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text) if s.strip()]


def make_chunks(text):
    """Group whole sentences into chunks of about CHUNK_TOKENS tokens, with a small overlap."""
    sentences = split_sentences(text)
    chunks, current, current_tokens = [], [], 0
    for sent in sentences:
        t = n_tokens(sent)
        if current and current_tokens + t > CHUNK_TOKENS:
            chunks.append(" ".join(current))
            # start the next chunk with the last few sentences (the overlap)
            overlap, overlap_tokens = [], 0
            for s in reversed(current):
                st = n_tokens(s)
                if overlap_tokens + st > OVERLAP_TOKENS:
                    break
                overlap.insert(0, s)
                overlap_tokens += st
            current, current_tokens = overlap, overlap_tokens
        current.append(sent)
        current_tokens += t
    if current and current_tokens >= MIN_CHUNK_TOKENS:
        chunks.append(" ".join(current))
    return chunks


# ---------------- labels (metadata) ----------------
def institution_label(value):
    v = str(value).lower()
    if "not sri" in v:
        return "not_sl"
    if v.startswith("national"):
        return "national"
    if "+" in v or "mixed" in v:
        return "mixed"
    if v.startswith("non-state"):
        return "non_state"
    if v.startswith("state"):
        return "state"
    return "mixed"


def discipline_label(group):
    g = str(group).lower()
    if "medical" in g:
        return "medical"
    if "allied" in g:
        return "allied_health"
    if g.startswith("management"):
        return "management"
    return "general"


def build_labels(row):
    """ChromaDB labels must be plain text or numbers (no lists, no empty values)."""
    topic_tags = str(row["Topic tag"])
    return {
        "doc_id": int(row["ID"]),
        "layer": str(row["Layer"]).strip()[0],              # "1", "2" or "3"
        "title": str(row["Title"]),
        "year": str(row["Year"]),
        "link": str(row["Link"]),
        "main_topic": topic_tags.split(";")[0].strip(),     # one short topic per chunk
        "topics": topic_tags,
        "institution": institution_label(row["Institution type"]),
        "discipline": discipline_label(row["Student group"]),
        "evidence_type": str(row["Evidence type"]),
    }


# ---------------- build ----------------
def build():
    meta = pd.read_excel(META_FILE, sheet_name="Documents").fillna("")
    meta = meta[meta["Status"] != "Drop"]

    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_or_create_collection(COLLECTION, embedding_function=embed_fn)

    missing, total_chunks = [], 0
    for _, row in meta.iterrows():
        doc_id = int(row["ID"])
        files = list(DOCS_DIR.glob(f"{doc_id}.*"))
        if not files:
            missing.append(doc_id)
            continue

        text = clean_text(read_doc(files[0]))
        if len(text) < 500:
            print(f"[warn] ID {doc_id}: very little text found. Is it a scanned PDF or a login page?")
            continue

        chunks = make_chunks(text)
        labels = build_labels(row)
        col.upsert(                                  # upsert = safe to re-run
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{**labels, "chunk_no": i} for i in range(len(chunks))],
        )
        total_chunks += len(chunks)
        print(f"ID {doc_id:>2}: {len(chunks):>3} chunks  ({files[0].name})")

    print(f"\nDone. {total_chunks} chunks stored in '{DB_DIR}'.")
    if missing:
        print(f"No file found in {DOCS_DIR}/ for IDs: {missing}")


# ---------------- search ----------------
def _query(col, question, layer, k, institution=None, discipline=None):
    conditions = [{"layer": layer}]
    if institution:
        # prefer matching institution type, but keep documents that apply to everyone
        conditions.append({"institution": {"$in": [institution, "mixed", "national", "not_sl"]}})
    if discipline:
        # hide medical / allied-health papers unless the student studies that
        conditions.append({"discipline": {"$in": ["general", discipline]}})
    where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
    res = col.query(query_texts=[question], where=where, n_results=k)
    return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))


def search(question, institution=None, discipline=None, k=3):
    """Returns the best chunks from each layer, using the student's profile to narrow results."""
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_collection(COLLECTION, embedding_function=embed_fn)

    results = {}
    for layer, name in [("1", "Sri Lankan context"), ("2", "Interventions"), ("3", "Local resources")]:
        hits = _query(col, question, layer, k, institution, discipline)
        if len(hits) < 2:                            # filters too tight, so relax them
            hits = _query(col, question, layer, k)
        results[name] = hits
    return results


def show(results):
    for layer_name, hits in results.items():
        print(f"\n=== {layer_name} ===")
        for cid, text, meta, dist in hits:
            print(f"- [{meta['doc_id']}] {meta['title']}  (distance {dist:.2f})")
            print(f"  {text[:220]}...")


# ---------------- command line ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("question")
    s.add_argument("--institution", choices=["state", "non_state"])
    s.add_argument("--discipline", help="e.g. medical, management, computing")
    s.add_argument("-k", type=int, default=3)
    args = p.parse_args()

    if args.cmd == "build":
        build()
    else:
        show(search(args.question, args.institution, args.discipline, args.k))
