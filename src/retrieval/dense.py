"""Dense (vector) search. Try it:  python -m src.retrieval.dense "how does a pod restart a container" """
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION, EMBED_MODEL, QDRANT_URL, QUERY_PREFIX

_model = None


def search(query: str, k: int = 5):
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    vec = _model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    client = QdrantClient(url=QDRANT_URL)
    return client.query_points(COLLECTION, query=vec.tolist(), limit=k).points


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "what is a pod"
    for p in search(q):
        print(f"\n{p.score:.3f}  {p.payload['section']}\n      {p.payload['url']}")
        print("      " + p.payload["text"][:200].replace("\n", " ") + "...")
