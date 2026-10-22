"""
Builds the three-layer RAG knowledge base :
parses the markdown source files in sample_docs/, splits them into the
chunks already marked with `## chunk_id:` headers, embeds each chunk with a
multilingual sentence-transformer, and stores them (+ metadata) in a local
ChromaDB collection.


Usage:
    python build_kb.py                     # (re)builds from sample_docs/
    python build_kb.py --source my_docs/    # build from a different folder
"""
import argparse
import re
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHROMA_DIR, KB_SOURCE_DIR  # noqa: E402

LAYER_BY_FILENAME = {
    "cultural_context.md": "cultural",
    "interventions.md": "intervention",
    "local_resources.md": "local",
}

CHUNK_HEADER_RE = re.compile(
    r"^## chunk_id:\s*(?P<chunk_id>\S+)\n"
    r"source:\s*(?P<source>[^\n]+)\n"     # [^\n]+ (not ".+") so DOTALL below
    r"tags:\s*(?P<tags>[^\n]+)\n"         # doesn't let these swallow the body
    r"(?P<body>.*?)(?=\n## chunk_id:|\Z)",
    re.DOTALL | re.MULTILINE,
)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # supports EN + Sinhala
COLLECTION_NAME = "workloadlens_kb"


def parse_markdown_file(path: Path, layer: str):
    text = path.read_text(encoding="utf-8")
    chunks = []
    for m in CHUNK_HEADER_RE.finditer(text):
        chunks.append({
            "id": m.group("chunk_id").strip(),
            "text": m.group("body").strip(),
            "metadata": {
                "layer": layer,
                "source": m.group("source").strip(),
                "tags": m.group("tags").strip(),
                "file": path.name,
            },
        })
    return chunks


def load_all_chunks(source_dir: Path):
    all_chunks = []
    for path in sorted(source_dir.glob("*.md")):
        layer = LAYER_BY_FILENAME.get(path.name, "uncategorised")
        all_chunks.extend(parse_markdown_file(path, layer))
    return all_chunks


def build(source_dir: Path):
    chunks = load_all_chunks(source_dir)
    if not chunks:
        raise ValueError(f"No chunks found in {source_dir}. Check your `## chunk_id:` headers.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads ~470MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # start fresh each build so stale chunks from a previous run don't linger
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks into ChromaDB collection "
          f"'{COLLECTION_NAME}' at {CHROMA_DIR}")
    for layer in ["cultural", "intervention", "local"]:
        n = sum(1 for c in chunks if c["metadata"]["layer"] == layer)
        print(f"  - {layer}: {n} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=str(KB_SOURCE_DIR))
    args = parser.parse_args()
    build(Path(args.source))
