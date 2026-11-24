"""Optional second-pass reranker (cross-encoder).

First-stage search (dense + BM25) is fast but rough. A cross-encoder reads the query and each
candidate *together* and scores how well they match: slower, but more precise. We only run it
on the ~20 fused candidates.
"""
from typing import Callable

from src.config import RERANK_MODEL

_model = None


def _predict(pairs: list[tuple[str, str]]) -> list[float]:
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL, max_length=512)
    return [float(s) for s in _model.predict(pairs)]


def rerank(query: str, records: list[dict], top_k: int = 10,
           predict_fn: Callable[[list[tuple[str, str]]], list[float]] | None = None) -> list[dict]:
    """Return the top_k records re-ordered by cross-encoder score (best first)."""
    if not records:
        return []
    scores = (predict_fn or _predict)([(query, r["text"]) for r in records])
    order = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order[:top_k]:
        rec = dict(records[i])
        rec["rerank_score"] = scores[i]
        out.append(rec)
    return out


def warmup() -> None:
    """Load the model once at startup so the first request isn't slow."""
    _predict([("warmup", "warmup")])
