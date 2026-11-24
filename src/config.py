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

# ---- LLM (any OpenAI-compatible chat API; defaults to Groq) ----
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")   # only sent to gpt-oss models
TOP_K = int(os.getenv("TOP_K", "5"))
# Cross-encoder reranking of the fused candidates: better ranking, but a few extra seconds on CPU.
# Which system prompt to use: "v1" (original) or "v2" (stricter about unsupported facts).
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
USE_RERANK = os.getenv("USE_RERANK", "1") == "1"

# ---- Evaluation / reranking ----
EVAL_DIR = ROOT / "eval"
TESTSET_PATH = EVAL_DIR / "testset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
# Small cross-encoder that is fast enough on CPU.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
