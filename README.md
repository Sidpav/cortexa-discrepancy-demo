# Cortexa Chunk-Based Gaps / Inconsistencies Demo

This Streamlit demo tests whether retrieved chunks alone are sufficient to identify discrepancies for a user query.

## What it does

1. User uploads one or more documents.
2. The app extracts text and chunks the documents.
3. The user asks a question.
4. The retriever returns the top K relevant chunks, default K = 10.
5. A Qwen 3B-class model reviews only those chunks.
6. The app displays:
   - user query
   - retrieved chunk ids
   - inconsistency id
   - 1-3 discrepancies ordered by severity
   - evidence from chunk ids
7. If no discrepancies are found, it returns:
   - original user prompt
   - “No discrepancies found”
8. Follow-up questions repeat the same process.
9. You can test different K values such as 10, 20, 30, 40, and 50 to see when meaningful discrepancies start appearing.

## Model

Default model through Transformers:

```bash
Qwen/Qwen2.5-3B-Instruct
```

Run with Transformers:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional Ollama backend:

```bash
export LLM_BACKEND=ollama
export OLLAMA_MODEL=qwen2.5:3b-instruct
streamlit run app.py
```

For UI testing without loading a model:

```bash
export LLM_BACKEND=mock
streamlit run app.py
```

## Notes for the EOD experiment

Start with:
- Top K = 10
- Chunk size = 260 words
- Overlap = 50 words

Then compare with:
- K = 20
- K = 30
- K = 50

Track:
- Whether discrepancies are found at each K
- Whether they are real or just period/scope differences
- Whether the evidence chunks are sufficient to explain the discrepancy
- Whether increasing K improves discrepancy quality or introduces noise

## Production caveat

This is a demo, not a full production system. For production, the Collector should pass stable chunk IDs, document names, page references, and retrieval scores directly into the prompt.
