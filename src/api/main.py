"""HTTP API for DocuMind.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    open http://localhost:8000/docs   (interactive documentation)
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import CHUNKS_PATH, COLLECTION, LLM_MODEL, QDRANT_URL, TOP_K, USE_RERANK
from src.generation.llm import LLMError
from src.generation.rag import answer as rag_answer
from src.generation.rewrite import contextualize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("documind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("DOCUMIND_WARMUP", "1") == "1":
        from src.retrieval.bm25 import get_index
        from src.retrieval.dense import warmup
        log.info("Warming up (BM25 index, embedding model, Qdrant)...")
        get_index()
        warmup()
        if USE_RERANK:
            from src.retrieval.rerank import warmup as rerank_warmup
            rerank_warmup()
        log.info("Ready.")
    yield


app = FastAPI(title="DocuMind", description="RAG question answering over the Kubernetes docs", lifespan=lifespan)


# ---------- request / response models ----------
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    k: int = Field(default=TOP_K, ge=1, le=10)
    rerank: bool | None = Field(default=None, description="Use the cross-encoder reranker. Omit to use the server default.")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Source(BaseModel):
    n: int
    section: str
    url: str


class Passage(Source):
    dense_rank: int | None = None
    bm25_rank: int | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    found: bool
    sources: list[Source]
    retrieved: list[Passage]
    rewritten_question: str | None = None
    reranked: bool = False
    timings: dict[str, float]
    usage: dict


# ---------- endpoints ----------
def _probe() -> dict:
    from qdrant_client import QdrantClient
    vectors = QdrantClient(url=QDRANT_URL).count(COLLECTION).count
    chunks = sum(1 for _ in open(CHUNKS_PATH, encoding="utf-8"))
    return {"vectors": vectors, "chunks": chunks}


@app.get("/health")
def health():
    try:
        info = _probe()
    except Exception as e:
        log.exception("health check failed")
        raise HTTPException(status_code=503, detail=f"Search backend unavailable: {type(e).__name__}")
    return {"status": "ok", "llm_model": LLM_MODEL, "rerank_default": USE_RERANK, **info}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t_start = time.time()
    standalone, rewrite_s = req.question, 0.0
    use_rerank = USE_RERANK if req.rerank is None else req.rerank

    if req.history:
        t = time.time()
        try:
            standalone = contextualize(req.question, [h.model_dump() for h in req.history])
        except Exception as e:                      # rewriting is a nice-to-have: fall back to the raw question
            log.warning("query rewrite failed (%s); using the original question", e)
        rewrite_s = time.time() - t

    try:
        res = rag_answer(standalone, k=req.k, rerank=use_rerank)
    except LLMError as e:
        log.error("LLM error: %s", e)
        raise HTTPException(status_code=502, detail=f"The language model call failed: {e}")
    except Exception as e:
        log.exception("retrieval failed")
        raise HTTPException(status_code=503, detail=f"Search backend unavailable: {type(e).__name__}")

    llm = res["llm"]
    total = time.time() - t_start
    log.info("q=%r standalone=%r found=%s tokens=%s+%s total=%.2fs",
             req.question[:80], standalone[:80], res["found"],
             llm.get("prompt_tokens"), llm.get("completion_tokens"), total)

    return AskResponse(
        answer=res["answer"],
        found=res["found"],
        sources=res["sources"],
        retrieved=[
            Passage(n=i, section=c["section"], url=c["url"], dense_rank=c.get("dense_rank"),
                    bm25_rank=c.get("bm25_rank"), text=c["text"][:1500])
            for i, c in enumerate(res["retrieved"], 1)
        ],
        rewritten_question=standalone if standalone != req.question else None,
        reranked=use_rerank,
        timings={"rewrite_s": round(rewrite_s, 2), "retrieval_s": res["retrieval_seconds"],
                 "generation_s": llm["seconds"], "total_s": round(total, 2)},
        usage={"model": llm["model"], "prompt_tokens": llm["prompt_tokens"],
               "completion_tokens": llm["completion_tokens"]},
    )
