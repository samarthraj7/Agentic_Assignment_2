# Process log

## AI tools used

Cursor (Claude / Grok 4.6) wrote the FastAPI + LangGraph supervisor, RAG ingest, vanilla SSE frontend, Dockerfile, and deploy script. Runtime LLMs: Gemini 3.8 Flash (supervisor / cheap lookups) and NVIDIA NIM `nvidia/nemotron-3-ultra-550b-a55b` (complex workers and fallback). Tavily Remote MCP is shared by research and summary. Local Python 3.14, pytest, and `python -m src.eval.ragas_eval --generate` (NVIDIA as judge). Homebrew and gcloud were setup only.

## Development narrative

I followed the rubric and Papa’s rules first, then the named stack: LangGraph supervisor, FastAPI + SSE, vanilla HTML, Chroma, Tavily Remote MCP. The project started inside Assignment 1; I moved it to its own repo so `csci599-a1` was never edited. NVIDIA hosted embeddings on this key are retired (410 Gone), so ingest uses local `fastembed` (`BAAI/bge-small-en-v1.5`) — 4,620 chunks, ~19 minutes. Deploy broke twice: a leading space in `GOOGLE_API_KEY=` made bash treat the key as a command, and `gcloud builds submit` followed `.gitignore` and uploaded 57 KB without `chroma_index/`. `.gcloudignore` fixed the second; the live image is 44.2 MiB. Gemini’s free tier is 20 requests/day; after the 429s the router’s NVIDIA fallback is what kept answers coming. Supervisor calls that ended on an AI message got Gemini 400 (“Requests ending with a model turn are not supported”); I now send only `[System, Human(query)]` plus worker findings. “What’s OpenClaw?” finished on the server in 146 s (Gemini 120 s hang, then Tavily + NVIDIA) but the UI stayed blank because workers use `ainvoke` and we only streamed `AIMessageChunk`. Fixes: 25 s Gemini timeout, flush the final worker text, Tavily first for live names, a “waiting for supervisor” status line, and the MCP key in `Authorization` not the URL.

## Verification narrative

Live `/health` lists research, code, summary, and Tavily tools. Assignment 1 `/health` still has filesystem, Tavily, and weather. A RAG question cited `checkpointers.mdx` / `persistence.mdx` (TTFT 4.5 s, total 48.7 s; retrieval 0.04 s warm, worker LLM 17.6 s). After the UI fix, OpenClaw answered in 40 s with `tavily_search`. Idle `/health` after scale-to-zero was 2.86 s; adding that to warm TTFT gives a first-chat cold TTFT of about 7–8 s on an already-built revision, or ~20–25 s if the instance has to boot (spec reference). pytest: 8 passed. RAGAS-style eval on the eight-question set: context precision 0.50, context recall 0.875, faithfulness 0.87, answer relevance 0.90 (NVIDIA judge; official RAGAS package not used because Gemini quota blocked the usual judge). I also sent a live-web query and a code-shaped query — not only the checkpointer prompt.

## What I learned

The tools did not catch bash `source` of an unquoted key, Cloud Build omitting the index because it honors `.gitignore`, Gemini 400 when the last message is an AI turn, or `ainvoke` leaving the SSE stream empty. Those showed up in Cloud Run logs and a blank chat bubble, not in the first generated draft. A two-provider router only counts if the fallback fires and the UI prints the result.
