"""Minimal client for any OpenAI-compatible chat API (Groq by default), with retry/backoff.

The free tier has per-minute request and token limits, so 429 responses are expected:
we wait for the server's Retry-After (or back off exponentially) and try again.
"""
import time

import requests

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_REASONING_EFFORT

RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    retries: int = 5,
    timeout: int = 90,
) -> dict:
    """Return {"text", "model", "prompt_tokens", "completion_tokens", "seconds"}."""
    if not LLM_API_KEY:
        raise LLMError("No API key found. Put GROQ_API_KEY=... in .env")
    model = model or LLM_MODEL
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_completion_tokens": max_tokens}
    if LLM_REASONING_EFFORT and "gpt-oss" in model:
        payload["reasoning_effort"] = LLM_REASONING_EFFORT   # reasoning models: keep thinking short

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    start = time.time()
    for attempt in range(retries):
        try:
            r = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:          # network hiccup: treat like a retryable error
            if attempt == retries - 1:
                raise LLMError(f"Network error: {e}") from e
            time.sleep(min(2 ** attempt, 30))
            continue

        if r.status_code == 200:
            data = r.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise LLMError("Empty answer (the model may have used all max_tokens thinking). "
                               "Try raising max_tokens or lowering LLM_REASONING_EFFORT.")
            usage = data.get("usage", {})
            return {"text": text, "model": model,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "seconds": round(time.time() - start, 2)}

        if r.status_code in RETRY_STATUS and attempt < retries - 1:
            try:
                wait = float(r.headers.get("retry-after", 2 ** attempt))
            except ValueError:
                wait = 2 ** attempt
            time.sleep(min(wait, 60))
            continue
        raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise LLMError("Gave up after repeated rate-limit/server errors")
