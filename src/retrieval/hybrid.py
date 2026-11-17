"""Hybrid retrieval: dense (meaning) + BM25 (keywords), merged with Reciprocal Rank Fusion.

    python -m src.retrieval.hybrid "kubectl rollout undo"
    python -m src.retrieval.hybrid "terminationGracePeriodSeconds" --compare
"""
import sys
from typing import Callable

RRF_K = 60  # standard constant; larger = flatter weighting of top ranks


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Each ranking is a list of ids, best first.
    score(id) = sum over rankings of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, start=1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _dense_ids(query: str, n: int) -> list[str]:
    from src.retrieval.dense import search as dense_search   # imported lazily: loads the embedding model
    return [str(p.id) for p in dense_search(query, n)]


def _bm25_ids(query: str, n: int) -> list[str]:
    from src.retrieval.bm25 import search as bm25_search
    return [r["id"] for r, _ in bm25_search(query, n)]


def hybrid_search(
    query: str,
    k: int = 5,
    n_candidates: int = 20,
    dense_fn: Callable[[str, int], list[str]] = _dense_ids,
    bm25_fn: Callable[[str, int], list[str]] = _bm25_ids,
) -> list[dict]:
    """Return the top-k fused chunks. Each result also records its rank in each retriever
    (None = not in that retriever's top n_candidates), which makes the behaviour easy to inspect."""
    from src.retrieval.bm25 import get_index
    by_id = {r["id"]: r for r in get_index().records}

    dense_ids, bm25_ids = dense_fn(query, n_candidates), bm25_fn(query, n_candidates)
    dense_rank = {i: r for r, i in enumerate(dense_ids, 1)}
    bm25_rank = {i: r for r, i in enumerate(bm25_ids, 1)}

    results = []
    for id_, score in rrf([dense_ids, bm25_ids])[:k]:
        rec = dict(by_id[id_])
        rec.update(score=score, dense_rank=dense_rank.get(id_), bm25_rank=bm25_rank.get(id_))
        results.append(rec)
    return results


def _short(rec: dict, n: int = 95) -> str:
    s = rec["section"]
    return s if len(s) <= n else "..." + s[-(n - 3):]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = " ".join(args) or "how does a pod restart a container"
    if "--compare" in sys.argv:
        from src.retrieval.bm25 import get_index
        by_id = {r["id"]: r for r in get_index().records}
        for name, ids in (("DENSE", _dense_ids(query, 5)), ("BM25", _bm25_ids(query, 5))):
            print(f"\n== {name}")
            for i, id_ in enumerate(ids, 1):
                print(f"  {i}. {_short(by_id[id_])}")
        print("\n== HYBRID (RRF)")
    for i, r in enumerate(hybrid_search(query), 1):
        print(f"  {i}. [dense#{r['dense_rank'] or '-'} bm25#{r['bm25_rank'] or '-'}] {_short(r)}")
        print(f"       {r['url']}")
