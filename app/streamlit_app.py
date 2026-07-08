"""Streamlit chat UI for the RAG Document Assistant API."""

import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")
st.title("📄 RAG Document Assistant")
st.caption("Upload PDFs, then ask questions. Every answer is grounded in cited sources.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Documents")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.button("Ingest", use_container_width=True):
        with st.spinner("Ingesting…"):
            resp = requests.post(
                f"{API_URL}/documents",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=300,
            )
        if resp.ok:
            st.success(f"Ingested {resp.json()['chunks_added']} chunks from {uploaded.name}")
        else:
            st.error(resp.json().get("detail", "Ingestion failed"))

    try:
        docs = requests.get(f"{API_URL}/documents", timeout=10).json()["documents"]
    except requests.RequestException:
        st.error(f"API not reachable at {API_URL}. Start it with `make run-api`.")
        docs = {}

    if docs:
        st.subheader("Ingested")
        for source, n_chunks in sorted(docs.items()):
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{source}** — {n_chunks} chunks")
            if col2.button("🗑", key=f"del-{source}", help=f"Delete {source}"):
                requests.delete(f"{API_URL}/documents/{source}", timeout=30)
                st.rerun()
    else:
        st.info("No documents yet — upload a PDF to get started.")

    top_k = st.slider("Sources per question (top-k)", 1, 15, 10)

# ---------------------------------------------------------------- chat
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        for citation in turn.get("citations", []):
            with st.expander(
                f"[{citation['source_index']}] {citation['source']}, p. {citation['page']}"
            ):
                st.markdown(f"> {citation['quote']}")

question = st.chat_input("Ask a question about your documents…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Thinking…"):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "top_k": top_k},
                timeout=300,
            )
            if resp.ok:
                data = resp.json()
                answer, citations = data["answer"], data["citations"]
                if not data["grounded"]:
                    answer = f"⚠️ *Not grounded in the documents.*\n\n{answer}"
            else:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:  # non-JSON body (e.g. a bare 500)
                    detail = resp.text or f"HTTP {resp.status_code}"
                answer = f"Error: {detail}"
                citations = []
        except requests.RequestException as exc:
            answer, citations = f"Error: could not reach the API ({exc}).", []

        st.markdown(answer)
        for citation in citations:
            with st.expander(
                f"[{citation['source_index']}] {citation['source']}, p. {citation['page']}"
            ):
                st.markdown(f"> {citation['quote']}")

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )
