"""Dense (vector) search. Try it:  python -m src.retrieval.dense "how does a pod restart a container" """
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION, EMBED_MODEL, QDRANT_URL, QUERY_PREFIX

_model = None
_client = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def warmup() -> None:
    """Load the model and connect once at startup so the first request isn't slow."""
    _get_model().encode(QUERY_PREFIX + "warmup", normalize_embeddings=True)
    _get_client().count(COLLECTION)


def search(query: str, k: int = 5):
    vec = _get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    return _get_client().query_points(COLLECTION, query=vec.tolist(), limit=k).points


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "what is a pod"
    for p in search(q):
        print(f"\n{p.score:.3f}  {p.payload['section']}\n      {p.payload['url']}")
        print("      " + p.payload["text"][:200].replace("\n", " ") + "...")
