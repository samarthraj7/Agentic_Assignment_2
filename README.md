# CSCI 599 Assignment 2 — Multi-Agent RAG

Supervisor-orchestrated LangGraph system: three specialized agents (research, code, summary) retrieve from a Chroma index of LangGraph / LangChain / FastAPI docs, share Tavily Remote MCP for live web knowledge, and route generation across **Google Gemini** and **NVIDIA NIM**. FastAPI streams tokens, traces, and citations over SSE to a single vanilla HTML page.

**Live URL:** https://multi-agent-rag-881242810047.us-west1.run.app

Assignment 1 is a different Cloud Run service in the same GCP project and was not changed: https://csci599-a1-4mbqu4lsoa-uw.a.run.app

## Architecture in one paragraph

A supervisor node classifies each turn and hands it to one worker at a time. Workers return to the supervisor, which either chains another worker or finishes. The research agent has `search_docs` plus Tavily search; the code agent has `search_code_examples` plus a local `check_python` AST tool; the summary agent has `search_docs`, `fetch_document`, Tavily search, and Tavily extract. The same Tavily MCP tools are wrapped per-agent so the trace shows which agent invoked them. LLM routing is hybrid: Gemini Flash for classification, summaries, and short lookups; Nemotron Ultra for multi-step synthesis. Failures (including Gemini free-tier 429s) retry on the other provider and appear in the trace.

See `docs/architecture.md` for the state schema, routing table, and RAG pipeline. See `docs/evaluation.md` for measured latency.

## Prerequisites

- Python 3.13 or 3.14
- `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) (free tier; Flash is 20 requests/day)
- `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com)
- `TAVILY_API_KEY` from [tavily.com](https://tavily.com) (student/free plan)

Embeddings run locally (`fastembed` / `BAAI/bge-small-en-v1.5`). No embedding API key is required; NVIDIA hosted embeddings on this account are retired.

## Local development

```bash
cd /Users/samarth_raj/Agentic_Assignment_2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill GOOGLE_API_KEY, NVIDIA_API_KEY, TAVILY_API_KEY
chmod +x scripts/fetch_corpus.sh
./scripts/fetch_corpus.sh
.venv/bin/python -m src.rag.ingest
.venv/bin/python main.py
```

Open http://127.0.0.1:8080. Conversation state is a local SQLite checkpointer keyed by `session_id`.

Quote values in `.env` (`GOOGLE_API_KEY="..."`). A leading space after `=` makes bash treat the key as a command when `deploy.sh` sources the file.

Rebuild the index after changing the corpus:

```bash
./scripts/fetch_corpus.sh
.venv/bin/python -m src.rag.ingest --reset
```

## Deploy

Same GCP project as Assignment 1 (`csci-599-agenticai`), **different** service name (`multi-agent-rag`). Do not run Assignment 1’s `deploy.sh`.

```bash
cd /Users/samarth_raj/Agentic_Assignment_2
export GCP_PROJECT_ID=csci-599-agenticai
export GCP_REGION=us-west1
export CLOUD_RUN_SERVICE=multi-agent-rag
./deploy.sh
```

`.gcloudignore` (not `.gitignore`) controls the Cloud Build upload so `chroma_index/` is included and `.env` is not. Image: Python 3.13-slim, non-root, 1 GiB, 300 s timeout, concurrency 10, max 3 instances. Secrets are Cloud Run env vars.

This deploy does **not** claim durable multi-turn memory across instance recycling. Local and Cloud Run use SQLite (`/tmp/state.sqlite` on Cloud Run is not durable). Firestore or GCS/GCSFuse would be required before advertising that.

## Cost / access

| Role | Provider | Model | Why |
|---|---|---|---|
| Supervisor | Google | `gemini-3.8-flash` | cheap classification; falls back to NVIDIA on 429 |
| Research / Code (simple) | Google | `gemini-3.8-flash` | short grounded lookup |
| Research / Code (complex) | NVIDIA | `nvidia/nemotron-3-ultra-550b-a55b` | multi-step synthesis |
| Summary | Google | `gemini-3.8-flash` | long-context, low cost |
| Embeddings | local | `BAAI/bge-small-en-v1.5` | no API cost |

Gemini 3.8 Flash free-tier generate-content is 20 requests/day on this project (hit during verification; router then used NVIDIA). Re-check [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and NVIDIA NIM terms before a long eval. Cloud Run compute is on the USC education credit.

## Tests

```bash
.venv/bin/pytest -q
```
