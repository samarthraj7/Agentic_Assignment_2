# Process log

*200–300 words, four sections. Draft; rewrite after the first working Cloud Run deploy.*

## AI tools used

Cursor (Claude) for scaffolding the FastAPI / LangGraph / RAG layout. Local Python 3.14 for ingestion and tests. Provider SDKs: `langchain-google-genai`, `langchain-openai` against NVIDIA NIM.

## Development narrative

Assignment 2 was moved out of the Assignment 1 folder into its own repo so A1 files, its git history, and its Cloud Run service stay untouched. The first NVIDIA embedding model I tried (`nv-embedqa-e5-v5`) and every other hosted NVIDIA embedding on this key are retired or disabled, so embeddings are local `fastembed`. Chroma ingestion of ~4.6k chunks is slow on a laptop (~30 s per 64-chunk batch) and is the current bottleneck. Supervisor + three workers, hybrid Gemini/NVIDIA routing, and shared Tavily MCP wrappers are written but not yet verified end-to-end.

## Verification narrative

*(Fill after: local `/health`, a RAG query, a live-web query, a code query, a cold-start Cloud Run request, and a query the system was not rehearsed on.)*

## What I learned

*(Fill after the first real failure the tools could not fix — likely the embedding-model retirement, SSE-over-POST vs EventSource, or Cloud Run cold-start / state.)*
