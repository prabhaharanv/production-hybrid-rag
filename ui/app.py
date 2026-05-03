"""Streamlit Web UI for Production Hybrid RAG.

Features:
- Real-time streaming responses via SSE
- Source panel showing retrieved documents with relevance scores
- Confidence indicators for answer quality
- Chat history with conversation context
"""

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
API_KEY = os.getenv("RAG_API_KEY", "")

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar: Configuration & Sources ----
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", value=API_URL)
    api_key = st.text_input("API Key", value=API_KEY, type="password")
    top_k = st.slider("Retrieved chunks (top_k)", min_value=1, max_value=20, value=5)
    use_streaming = st.toggle("Stream responses", value=True)

    st.divider()
    st.title("📚 Sources")
    sources_container = st.container()

# ---- Main area ----
st.title("🔍 Production Hybrid RAG")
st.caption("Ask questions about your documents — answers are grounded in retrieved context with citations.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metadata"):
            _display_confidence(msg["metadata"])


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _display_confidence(metadata: dict):
    """Show confidence indicators based on response metadata."""
    col1, col2, col3 = st.columns(3)

    abstained = metadata.get("abstained", False)
    num_citations = len(metadata.get("citations", []))
    chunks = metadata.get("retrieved_chunks", [])
    avg_score = sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1)

    with col1:
        if abstained:
            st.metric("Confidence", "Low", delta="Abstained", delta_color="inverse")
        elif avg_score > 0.7:
            st.metric("Confidence", "High", delta=f"{avg_score:.0%}")
        elif avg_score > 0.4:
            st.metric("Confidence", "Medium", delta=f"{avg_score:.0%}")
        else:
            st.metric("Confidence", "Low", delta=f"{avg_score:.0%}", delta_color="inverse")

    with col2:
        st.metric("Citations", num_citations)

    with col3:
        st.metric("Sources", len(chunks))


def _display_sources(chunks: list[dict], citations: list[dict]):
    """Render the source panel in the sidebar."""
    with sources_container:
        sources_container.empty()
        if not chunks:
            st.info("No sources retrieved.")
            return

        cited_refs = {c["reference"] for c in citations}

        for i, chunk in enumerate(chunks, 1):
            cited = "✅" if i in cited_refs else ""
            score = chunk.get("score", 0)
            score_bar = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"

            with st.expander(f"{score_bar} [{i}] {chunk.get('title', 'Unknown')} {cited}", expanded=(i in cited_refs)):
                st.caption(f"Score: {score:.3f} | Source: `{chunk.get('source', 'N/A')}`")
                st.text(chunk.get("text", "")[:500])


def _ask_streaming(question: str, top_k: int):
    """Call /ask/stream and render tokens in real-time."""
    response = requests.post(
        f"{api_url}/ask/stream",
        headers=_build_headers(),
        json={"question": question, "top_k": top_k},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    answer_tokens = []
    metadata = {}
    placeholder = st.empty()

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        event = payload.get("event")

        if event == "metadata":
            metadata["rewritten_query"] = payload.get("rewritten_query", "")
            metadata["retrieved_chunks"] = payload.get("retrieved_chunks", [])
            _display_sources(metadata["retrieved_chunks"], [])

        elif event == "token":
            answer_tokens.append(payload["data"])
            placeholder.markdown("".join(answer_tokens) + "▌")

        elif event == "done":
            metadata["abstained"] = payload.get("abstained", False)
            metadata["citations"] = payload.get("citations", [])

    final_answer = "".join(answer_tokens)
    if metadata.get("abstained"):
        final_answer = "I don't have enough information in the available documents to answer this question."

    placeholder.markdown(final_answer)
    _display_confidence(metadata)
    _display_sources(metadata.get("retrieved_chunks", []), metadata.get("citations", []))
    return final_answer, metadata


def _ask_sync(question: str, top_k: int):
    """Call /ask (non-streaming) and display result."""
    response = requests.post(
        f"{api_url}/ask",
        headers=_build_headers(),
        json={"question": question, "top_k": top_k},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    answer = data["answer"]
    metadata = {
        "rewritten_query": data.get("rewritten_query", ""),
        "abstained": data.get("abstained", False),
        "citations": data.get("citations", []),
        "retrieved_chunks": data.get("retrieved_chunks", []),
    }

    st.markdown(answer)
    _display_confidence(metadata)
    _display_sources(metadata["retrieved_chunks"], metadata["citations"])
    return answer, metadata


# ---- Chat input ----
if question := st.chat_input("Ask a question about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Show assistant response
    with st.chat_message("assistant"):
        try:
            if use_streaming:
                answer, metadata = _ask_streaming(question, top_k)
            else:
                answer, metadata = _ask_sync(question, top_k)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata,
            })
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {api_url}. Is the server running?")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.status_code} — {e.response.text}")
