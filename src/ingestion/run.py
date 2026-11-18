"""Ingestion pipeline: load -> clean -> chunk -> embed -> store.

Run from the project root:   python -m src.ingestion.run            (incremental)
                              python -m src.ingestion.run --rebuild   (start from scratch)
"""
import hashlib
import json
import sys
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.config import (CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNKS_PATH,
                        COLLECTION, EMBED_DIM, EMBED_MODEL, QDRANT_URL)
from src.ingestion.chunker import chunk_page
from src.ingestion.cleaner import clean_markdown
from src.ingestion.loader import load_docs


def build_chunks() -> list[dict]:
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    tok.model_max_length = int(1e9)   # we only count tokens here; silences the harmless "1780 > 512" warning
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
                "hash": hashlib.sha1(c.text.encode("utf-8")).hexdigest(),
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


def _existing_hashes(client: QdrantClient) -> dict[str, str | None]:
    """id -> content hash for every point already in the collection."""
    found, offset = {}, None
    while True:
        points, offset = client.scroll(COLLECTION, limit=1000, offset=offset,
                                       with_payload=["hash", "text"], with_vectors=False)
        for p in points:
            payload = p.payload or {}
            # Points from the first run have no stored hash: derive it from their stored text.
            h = payload.get("hash") or (
                hashlib.sha1(payload["text"].encode("utf-8")).hexdigest() if payload.get("text") else None)
            found[str(p.id)] = h
        if offset is None:
            return found


def store(records: list[dict], rebuild: bool = False) -> None:
    """Embed only new or changed chunks; delete chunks that no longer exist."""
    client = QdrantClient(url=QDRANT_URL)
    if rebuild and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))

    existing = _existing_hashes(client)
    todo = [r for r in records if existing.get(r["id"]) != r["hash"]]
    stale = list(set(existing) - {r["id"] for r in records})
    print(f"Unchanged: {len(records) - len(todo)}  To embed: {len(todo)}  To delete: {len(stale)}")
    if stale:
        client.delete(COLLECTION, points_selector=PointIdsList(points=stale))
    if not todo:
        print("Nothing to embed.")
        return

    model = SentenceTransformer(EMBED_MODEL)
    start, batch = time.time(), 64
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vecs = model.encode([r["text"] for r in part], normalize_embeddings=True, batch_size=32)
        client.upsert(COLLECTION, points=[
            PointStruct(id=r["id"], vector=v.tolist(),
                        payload={k: r[k] for k in ("text", "title", "section", "url", "path", "chunk_idx", "hash")})
            for r, v in zip(part, vecs)
        ])
        print(f"  embedded {min(i + batch, len(todo))}/{len(todo)}  ({time.time() - start:.0f}s)", flush=True)
    print(f"Collection '{COLLECTION}' now holds {client.count(COLLECTION).count} vectors")


if __name__ == "__main__":
    recs = build_chunks()
    save_jsonl(recs)
    print_stats(recs)
    store(recs, rebuild="--rebuild" in sys.argv)
