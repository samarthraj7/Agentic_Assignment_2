# CSCI 599 Assignment 2 — Multi-Agent RAG

Supervisor-orchestrated LangGraph system: three specialized agents (research, code, summary) retrieve from a Chroma index of LangGraph / LangChain / FastAPI docs, share Tavily Remote MCP for live web knowledge, and route generation across **Google Gemini** and **NVIDIA NIM**. FastAPI streams tokens, traces, and citations over SSE to a single vanilla HTML page.

Live URL: *(fill in after Cloud Run deploy)*

## Architecture in one paragraph

A supervisor node classifies each turn and hands it to one worker at a time. Workers return to the supervisor, which either chains another worker or finishes. The research agent has `search_docs` plus Tavily search; the code agent has `search_code_examples` plus a local `check_python` AST tool; the summary agent has `search_docs`, `fetch_document`, Tavily search, and Tavily extract. The same Tavily MCP tools are wrapped per-agent so the trace shows which agent invoked them. LLM routing is hybrid: Gemini Flash for classification, summaries, and short lookups; Nemotron Ultra for multi-step synthesis. Failures retry on the other provider.

See `docs/architecture.md` for the state schema, routing table, and RAG pipeline.

## Prerequisites

- Python 3.13 or 3.14
- `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) (free)
- `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com)
- `TAVILY_API_KEY` from [tavily.com](https://tavily.com) (student/free plan)

Embeddings run locally (`fastembed` / `BAAI/bge-small-en-v1.5`). No embedding API key is required; NVIDIA's hosted embedding models on this account are retired.

## Local development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill the three keys
chmod +x scripts/fetch_corpus.sh
./scripts/fetch_corpus.sh
.venv/bin/python -m src.rag.ingest
.venv/bin/python main.py
```

Open http://127.0.0.1:8080. Conversation state is a local SQLite checkpointer keyed by `session_id`.

Rebuild the index after changing the corpus:

```bash
./scripts/fetch_corpus.sh
.venv/bin/python -m src.rag.ingest --reset
```

## Deploy

```bash
export GCP_PROJECT_ID=your-project
./deploy.sh
```

The image is Python 3.13-slim, non-root, 1 GiB, 300 s timeout, concurrency 10, max 3 instances. The Chroma index is copied into the image; secrets go in as Cloud Run environment variables, never in source.

If you later claim durable multi-turn memory across Cloud Run instance recycling, switch `STATE_BACKEND` to Firestore (or GCS/GCSFuse). A single-turn or single-instance SQLite file under `/tmp` is not durable.

## Cost / access

| Role | Provider | Model | Why |
|---|---|---|---|
| Supervisor | Google | `gemini-3.8-flash` | cheap classification |
| Research / Code (simple) | Google | `gemini-3.8-flash` | short grounded lookup |
| Research / Code (complex) | NVIDIA | `nvidia/nemotron-3-ultra-550b-a55b` | multi-step synthesis |
| Summary | Google | `gemini-3.8-flash` | long-context, low cost |
| Embeddings | local | `BAAI/bge-small-en-v1.5` | no API cost |

Verify current Gemini and NVIDIA pricing before a long eval run. Cloud Run compute on the USC education credit covers the service itself.

## Tests

```bash
.venv/bin/pytest -q
```
