"""Keyword (BM25) search over the chunks saved by the ingestion step.

Plain BM25 tokenizers fall apart on Kubernetes text: `terminationGracePeriodSeconds`,
`--to-revision` and `kubectl.kubernetes.io/last-applied` are single "words" there.
Our tokenizer keeps the whole identifier AND its parts, so both a full-identifier
query and a loose natural-language query can match.
"""
import heapq
import json
import re

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "to", "of", "in", "on", "for", "and", "or",
    "it", "this", "that", "with", "as", "by", "be", "do", "does", "how", "what", "can", "i",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in _WORD.findall(text):
        whole = word.lower()
        parts = [p.lower() for p in _CAMEL_PART.findall(word.replace("_", " "))]
        if whole not in STOPWORDS and len(whole) > 1:
            tokens.append(whole)
        if len(parts) > 1:                       # camelCase / snake_case: add the parts too
            tokens.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    return tokens


class BM25Index:
    def __init__(self, records: list[dict]):
        self.records = records
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in records])

    def search(self, query: str, k: int = 20) -> list[tuple[dict, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        top = heapq.nlargest(k, range(len(scores)), key=scores.__getitem__)
        return [(self.records[i], float(scores[i])) for i in top if scores[i] > 0]


def load_records(path=CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


_index: BM25Index | None = None


def get_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index(load_records())
    return _index


def search(query: str, k: int = 20) -> list[tuple[dict, float]]:
    return get_index().search(query, k)
