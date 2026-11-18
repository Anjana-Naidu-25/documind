"""Turn a follow-up question into a standalone one, so retrieval works without the chat history.

"and how do I undo it?"  (after a question about deployments)
    ->  "How do I undo a Kubernetes Deployment rollout?"
"""
from typing import Callable

REWRITE_SYSTEM = """You rewrite follow-up questions for a Kubernetes documentation search system.
Given a conversation and the user's final question, output ONE standalone question that can be understood without the conversation: resolve words like "it", "that" or "the previous one", and keep exact resource names, flags and commands.
If the final question is already standalone, output it unchanged.
Output only the question, with no explanation."""

MAX_TURNS = 6
MAX_CHARS_PER_TURN = 500


def contextualize(question: str, history: list[dict], llm: Callable | None = None) -> str:
    if not history:
        return question
    if llm is None:
        from src.config import JUDGE_MODEL
        from src.generation.llm import chat
        llm = lambda msgs: chat(msgs, model=JUDGE_MODEL, max_tokens=400)

    convo = "\n".join(
        f"{t['role'].capitalize()}: {t['content'][:MAX_CHARS_PER_TURN]}" for t in history[-MAX_TURNS:]
    )
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nFinal question: {question}\n\nStandalone question:"},
    ]
    text = llm(messages)["text"].strip()
    first_line = text.splitlines()[0].strip().strip('"') if text else ""
    # Fall back to the original if the model returned nothing useful or something absurdly long.
    return first_line if 0 < len(first_line) <= 400 else question
