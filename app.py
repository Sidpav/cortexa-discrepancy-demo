from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from document_processing import chunk_documents, read_uploaded_file
from llm import QwenClient, parse_json_response, sort_and_limit_discrepancies
from prompts import DISCREPANCY_SYSTEM_PROMPT, build_discrepancy_prompt
from retrieval import ChunkRetriever


st.set_page_config(page_title="Cortexa Chunk Discrepancy Demo", layout="wide")

st.title("Cortexa Chunk-Based Gaps / Inconsistencies Demo")
st.caption("Upload documents, ask a question, retrieve chunks, and use a Qwen 3B-class model to find 1-3 material discrepancies.")

with st.sidebar:
    st.header("Demo Settings")
    top_k = st.slider("Chunks to retrieve", min_value=5, max_value=50, value=10, step=5)
    experiment_counts = st.multiselect(
        "Experiment with chunk counts",
        options=[10, 15, 20, 30, 40, 50],
        default=[10],
        help="Run the same query with different retrieval counts to test whether chunks alone are sufficient.",
    )
    chunk_words = st.slider("Chunk size", min_value=120, max_value=500, value=260, step=20)
    overlap_words = st.slider("Chunk overlap", min_value=0, max_value=150, value=50, step=10)
    st.divider()
    st.markdown("**Model**")
    st.write("Default: `Qwen/Qwen2.5-3B-Instruct` via Transformers")
    st.write("Set `LLM_BACKEND=ollama` to use local Ollama instead.")

uploaded_files = st.file_uploader(
    "Upload one or more documents",
    type=["pdf", "docx", "txt", "csv", "xlsx"],
    accept_multiple_files=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_files:
    with st.spinner("Reading and chunking documents..."):
        raw_docs = []
        for file in uploaded_files:
            raw_docs.extend(read_uploaded_file(file))
        chunks = chunk_documents(raw_docs, chunk_words=chunk_words, overlap_words=overlap_words)
        st.session_state.chunks = chunks
        st.session_state.retriever = ChunkRetriever(chunks) if chunks else None

    st.success(f"Loaded {len(uploaded_files)} file(s), created {len(chunks)} chunks.")

    with st.expander("Preview chunks"):
        preview = pd.DataFrame([
            {"chunk_id": c.chunk_id, "source": c.source, "page": c.page, "text_preview": c.text[:220] + "..."}
            for c in chunks[:25]
        ])
        st.dataframe(preview, use_container_width=True)
else:
    st.info("Upload documents to begin.")

st.divider()
query = st.text_area("User query", placeholder="Example: Are there any inconsistencies in HDFC Bank's valuation, recommendation, or financial metrics?")
run = st.button("Find discrepancies", type="primary", disabled=not uploaded_files or not query.strip())


def run_one_query(user_query: str, k: int) -> dict:
    retrieved = st.session_state.retriever.retrieve(user_query, top_k=k)
    user_prompt = build_discrepancy_prompt(user_query, retrieved)
    qwen = QwenClient()
    raw = qwen.generate(DISCREPANCY_SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_response(raw)
    parsed["user_query"] = user_query
    parsed["retrieved_chunk_ids"] = [c["chunk_id"] for c in retrieved]
    parsed["retrieved_chunks"] = retrieved
    parsed["top_k"] = k
    parsed["raw_model_output"] = raw
    return sort_and_limit_discrepancies(parsed, limit=3)


if run:
    counts = experiment_counts or [top_k]
    if top_k not in counts:
        counts = [top_k] + counts
    counts = sorted(set(counts))

    results = []
    for k in counts:
        with st.spinner(f"Running discrepancy check with top {k} chunks..."):
            result = run_one_query(query.strip(), k)
            results.append(result)

    st.session_state.history.append({"query": query.strip(), "results": results, "time": datetime.now().isoformat(timespec="seconds")})

if st.session_state.history:
    latest = st.session_state.history[-1]
    st.subheader("Latest Result")

    summary_rows = []
    for result in latest["results"]:
        severities = [d.get("severity", "") for d in result.get("discrepancies", [])]
        summary_rows.append({
            "Top K Chunks": result["top_k"],
            "Retrieved Chunk IDs": ", ".join(result.get("retrieved_chunk_ids", [])),
            "Discrepancies Found": "Yes" if result.get("discrepancies_found") else "No",
            "Count Displayed": len(result.get("discrepancies", [])),
            "Severities": ", ".join(severities) if severities else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    for result in latest["results"]:
        st.markdown(f"### Top {result['top_k']} chunk result")
        st.markdown(f"**User query:** {result['user_query']}")
        st.markdown(f"**Chunk IDs:** {', '.join(result.get('retrieved_chunk_ids', []))}")

        if not result.get("discrepancies_found"):
            st.success(f"{result['user_query']} — No discrepancies found")
        else:
            for d in result.get("discrepancies", []):
                severity = d.get("severity", "Medium")
                if severity in {"Critical", "High"}:
                    st.error(f"{d.get('inconsistency_id')} — {severity}: {d.get('title')}")
                elif severity == "Medium":
                    st.warning(f"{d.get('inconsistency_id')} — {severity}: {d.get('title')}")
                else:
                    st.info(f"{d.get('inconsistency_id')} — {severity}: {d.get('title')}")

                st.write(d.get("nicely_articulated_discrepancy", ""))
                ev = d.get("evidence", [])
                if ev:
                    st.markdown("**Evidence**")
                    st.dataframe(pd.DataFrame(ev), use_container_width=True)
                st.markdown(f"**Suggested resolution:** {d.get('suggested_resolution', '')}")

        with st.expander(f"Retrieved chunks for top {result['top_k']}"):
            chunk_df = pd.DataFrame([
                {
                    "chunk_id": c["chunk_id"],
                    "score": round(c.get("score", 0), 4),
                    "source": c.get("source"),
                    "page": c.get("page"),
                    "text": c.get("text")[:700] + "...",
                }
                for c in result.get("retrieved_chunks", [])
            ])
            st.dataframe(chunk_df, use_container_width=True)

        with st.expander(f"Raw model JSON for top {result['top_k']}"):
            st.code(json.dumps({k: v for k, v in result.items() if k not in {"retrieved_chunks", "raw_model_output"}}, indent=2), language="json")
            st.text_area("Raw Qwen output", value=result.get("raw_model_output", ""), height=180)

st.divider()
with st.expander("Conversation history"):
    for item in reversed(st.session_state.history):
        st.markdown(f"**{item['time']}** — {item['query']}")
