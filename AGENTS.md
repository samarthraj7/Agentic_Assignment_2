# CSCI 599 Assignment 2 — working notes

This repository is Assignment 2 only. Do not edit files under `/Users/samarth_raj/Agentic` (Assignment 1).

## Shape the grader will look for

- FastAPI backend, `POST /api/chat` with SSE events `token`, `trace`, `citation`, `error`, `done`.
- Vanilla HTML/CSS/JS frontend in `static/`. No React, Vue, Angular, Svelte, or `node_modules`.
- LangGraph supervisor with three specialized workers: research, code, summary.
- Two LLM providers: Gemini (cheap/fast) and NVIDIA NIM (capable). Routing reasons must appear in the UI.
- RAG over the LangGraph / LangChain / FastAPI docs in Chroma. Citations must survive into the answer.
- Tavily Remote MCP shared by the research and summary agents (MCP tool-sharing path of criterion 6).
- Pin exact versions in `requirements.txt`. Never commit `.env`.

## Local run

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill keys
./scripts/fetch_corpus.sh && .venv/bin/python -m src.rag.ingest
.venv/bin/python main.py
```

## What not to add

- A frontend framework or a Node build step.
- A second MCP server just to match Assignment 1 (filesystem / weather are not required here).
- A Firestore checkpointer until Cloud Run deploy time; local uses SQLite.
- Any change to Assignment 1 files, its `.gitignore`, or its Cloud Run service `csci599-a1`.
