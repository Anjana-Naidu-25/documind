"""Retrieval metrics. `ranked` is a list of ids/urls, best first; `relevant` is the set that counts as correct."""


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """1/rank of the first relevant result within the top k (0 if none)."""
    for i, r in enumerate(ranked[:k], 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def coverage_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top k (for questions needing several pages)."""
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
