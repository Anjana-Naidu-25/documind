from src.generation.rewrite import contextualize

HISTORY = [
    {"role": "user", "content": "How do I roll back a deployment?"},
    {"role": "assistant", "content": "Run kubectl rollout undo deployment/nginx-deployment [1]."},
]


def fake(text, calls=None):
    def _llm(messages):
        if calls is not None:
            calls.append(messages)
        return {"text": text}
    return _llm


def test_no_history_means_no_llm_call():
    calls = []
    assert contextualize("what is a pod?", [], llm=fake("x", calls)) == "what is a pod?"
    assert calls == []


def test_follow_up_is_rewritten():
    calls = []
    out = contextualize("and how do I check its history?", HISTORY,
                        llm=fake('"How do I check the rollout history of a Deployment?"', calls))
    assert out == "How do I check the rollout history of a Deployment?"
    prompt = calls[0][1]["content"]
    assert "How do I roll back a deployment?" in prompt and "Final question: and how do I check its history?" in prompt


def test_only_first_line_is_used():
    assert contextualize("q", HISTORY, llm=fake("Standalone question?\nHere is my explanation.")) == "Standalone question?"


def test_empty_or_absurd_output_falls_back():
    assert contextualize("original?", HISTORY, llm=fake("")) == "original?"
    assert contextualize("original?", HISTORY, llm=fake("x" * 500)) == "original?"


def test_history_is_truncated():
    calls = []
    long_history = [{"role": "user", "content": "z" * 5000}] * 20
    contextualize("q", long_history, llm=fake("ok?", calls))
    prompt = calls[0][1]["content"]
    assert prompt.count("User: ") == 6 and "z" * 501 not in prompt
