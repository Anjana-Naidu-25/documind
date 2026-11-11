from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # downloads on first run
client = QdrantClient(url="http://localhost:6333")

name = "smoke_test"
if client.collection_exists(name):
    client.delete_collection(name)
client.create_collection(
    name, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

docs = [
    "Employees are entitled to 24 days of annual leave per year.",
    "The office cafeteria serves lunch from 12 to 2 PM.",
    "Passwords must be rotated every 90 days.",
]
vecs = model.encode(docs, normalize_embeddings=True)
client.upsert(
    name,
    points=[
        PointStruct(id=i, vector=v.tolist(), payload={"text": d})
        for i, (d, v) in enumerate(zip(docs, vecs))
    ],
)

q = model.encode("How many days off do I get?", normalize_embeddings=True)
res = client.query_points(name, query=q.tolist(), limit=3)
for p in res.points:
    print(round(p.score, 3), p.payload["text"])
