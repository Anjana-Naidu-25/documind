"""Streamlit chat UI for DocuMind.   streamlit run ui/app.py"""
import os

import requests
import streamlit as st

DEFAULT_API = os.getenv("DOCUMIND_API", "http://localhost:8000")
EXAMPLES = [
    "How do I roll back a deployment?",
    "What does terminationGracePeriodSeconds do?",
    "What is the difference between a Deployment and a StatefulSet?",
    "How do I install Docker on Windows?",   # out of scope: should be refused
]

st.set_page_config(page_title="DocuMind", layout="wide")
st.title("DocuMind")
st.caption("Ask questions about the Kubernetes documentation. Every answer cites its sources, "
           "and says so when the documentation doesn't cover the question.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", DEFAULT_API).rstrip("/")
    k = st.slider("Passages to retrieve", 1, 10, 5)
    show_passages = st.toggle("Show retrieved passages", value=False)

    try:
        h = requests.get(f"{api_url}/health", timeout=3)
        if h.status_code == 200:
            info = h.json()
            st.success(f"API online · {info['vectors']} passages indexed")
        else:
            st.error(h.json().get("detail", "API unhealthy"))
    except requests.RequestException:
        st.error("API not reachable. Start it with: uvicorn src.api.main:app --port 8000")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Try one")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex-{ex}"):
            st.session_state.pending = ex


def render_meta(meta: dict) -> None:
    if meta.get("rewritten_question"):
        st.caption(f"Searched for: *{meta['rewritten_question']}*")
    if meta.get("sources"):
        st.markdown("**Sources**")
        for s in meta["sources"]:
            st.markdown(f"- [{s['n']}] [{s['section']}]({s['url']})")
    if show_passages and meta.get("retrieved"):
        with st.expander(f"Retrieved passages ({len(meta['retrieved'])})"):
            for p in meta["retrieved"]:
                ranks = f"dense #{p['dense_rank'] or '-'} · bm25 #{p['bm25_rank'] or '-'}"
                st.markdown(f"**[{p['n']}]** {p['section']}  \n`{ranks}` · [{p['url']}]({p['url']})")
                st.code(p["text"][:1000], language="markdown")
    t, u = meta.get("timings", {}), meta.get("usage", {})
    if t:
        st.caption(f"{u.get('model', '')} · retrieval {t['retrieval_s']}s · generation {t['generation_s']}s · "
                   f"{u.get('prompt_tokens')}+{u.get('completion_tokens')} tokens")


# ---------------- conversation ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_meta(m.get("meta", {}))

prompt = st.chat_input("Ask about Kubernetes...") or st.session_state.pop("pending", None)
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the docs..."):
            try:
                r = requests.post(f"{api_url}/ask", json={"question": prompt, "history": history, "k": k}, timeout=120)
            except requests.RequestException:
                r = None
        if r is None:
            reply, meta = "I couldn't reach the API. Is it running?", {}
            st.error(reply)
        elif r.status_code != 200:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
            reply, meta = f"Something went wrong: {detail}", {}
            st.error(reply)
        else:
            meta = r.json()
            reply = meta["answer"]
            st.markdown(reply)
            render_meta(meta)
    st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
