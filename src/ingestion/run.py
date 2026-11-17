"""Ingestion pipeline: load -> clean -> chunk -> embed -> store.

Run from the project root:   python -m src.ingestion.run
"""
import json
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.config import (CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNKS_PATH,
                        COLLECTION, EMBED_DIM, EMBED_MODEL, QDRANT_URL)
from src.ingestion.chunker import chunk_page
from src.ingestion.cleaner import clean_markdown
from src.ingestion.loader import load_docs


def build_chunks() -> list[dict]:
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    count = lambda s: len(tok.encode(s, add_special_tokens=False))

    records = []
    docs = load_docs()
    for doc in docs:
        meta, body = clean_markdown(doc.text)
        title = meta.get("title") or doc.rel_path
        for c in chunk_page(title, body, count, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS):
            records.append({
                # Deterministic id: re-running updates instead of duplicating.
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.rel_path}#{c.chunk_idx}")),
                "text": c.text,
                "title": title,
                "section": c.section,
                "url": doc.url,
                "path": doc.rel_path,
                "chunk_idx": c.chunk_idx,
                "tokens": count(c.text),
            })
    print(f"Pages: {len(docs)}  Chunks: {len(records)}")
    return records


def save_jsonl(records: list[dict]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {CHUNKS_PATH}")


def print_stats(records: list[dict]) -> None:
    t = sorted(r["tokens"] for r in records)
    print(f"Tokens per chunk: min={t[0]}  median={t[len(t)//2]}  "
          f"p95={t[int(len(t)*0.95)]}  max={t[-1]}  (model limit 512)")


def store(records: list[dict]) -> None:
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))

    start, batch = time.time(), 64
    for i in range(0, len(records), batch):
        part = records[i:i + batch]
        vecs = model.encode([r["text"] for r in part], normalize_embeddings=True, batch_size=32)
        client.upsert(COLLECTION, points=[
            PointStruct(id=r["id"], vector=v.tolist(),
                        payload={k: r[k] for k in ("text", "title", "section", "url", "path", "chunk_idx")})
            for r, v in zip(part, vecs)
        ])
        print(f"  embedded {min(i + batch, len(records))}/{len(records)}  ({time.time() - start:.0f}s)", flush=True)
    print(f"Stored {client.count(COLLECTION).count} vectors in '{COLLECTION}'")


if __name__ == "__main__":
    recs = build_chunks()
    save_jsonl(recs)
    print_stats(recs)
    store(recs)
