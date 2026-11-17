"""Central settings. Everything tunable lives here."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw" / "k8s"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

SITE_BASE_URL = "https://kubernetes.io/docs/"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
# bge models work best when the *query* (not the documents) has this prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "k8s_docs"

# bge-small truncates input at 512 tokens, so chunks must stay below that.
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "350"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))
