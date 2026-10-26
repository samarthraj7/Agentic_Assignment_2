# Process log

## AI tools used

- Cursor (Claude / Grok 4.6) — FastAPI + LangGraph supervisor, RAG ingest, SSE frontend, deploy script, this log
- NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`) — capable worker / fallback LLM
- Google Gemini 3.8 Flash — cheap supervisor and short lookups (free tier, 20 req/day)
- Tavily Remote MCP — live web search, shared by research and summary
- Homebrew / gcloud — machine setup, not code generation

## Development narrative

I read Assignment 2 the same way as Assignment 1: rubric first (25/20/15/15/15/10), Papa’s rules (a router that always picks one model is not a router; localhost is not a submission), then the stack the PDF names (LangGraph supervisor, FastAPI + SSE, vanilla HTML, Chroma or Pinecone, Tavily Remote MCP).

Assignment 2 started as a folder inside Assignment 1. I moved it to `/Users/samarth_raj/Agentic_Assignment_2` so A1’s files, git history, and Cloud Run service `csci599-a1` were never edited. NVIDIA’s hosted embedding models on this key are retired (410 Gone on `nv-embedqa-e5-v5` and the rest of the catalog), so ingest uses local `fastembed` (`BAAI/bge-small-en-v1.5`) into Chroma — 4,620 chunks, about 19 minutes. The first `./deploy.sh` failed with `command not found`. The next upload was 57 KB and would have shipped an empty index. The live service is `multi-agent-rag` in `csci-599-agenticai`; A1 is still `csci599-a1`.

## Debugging log — what went wrong and how I fixed it

Each entry is: what I saw, the evidence, and the fix. No keys are shown.

### 1. `./deploy.sh` → `command not found`

`.env` line 3 was `GOOGLE_API_KEY= <key>` (leading space). Bash `source` treats `VAR= value` as “run `value` with VAR empty”:

```
.env: line 3: <key>: command not found
```

Fix: quote every value (`GOOGLE_API_KEY="..."`) and source `.env` before reading `GCP_PROJECT_ID`. `deploy.sh` also prepends `/opt/homebrew/bin` so `gcloud` resolves in a Cursor terminal.

### 2. Cloud Build uploaded 57.6 KB and dropped the index

`gcloud builds submit` uses `.gitignore` when `.gcloudignore` is missing. `chroma_index/` is gitignored (rebuildable), so the first image had no vectors. Fix: `.gcloudignore` that excludes `.env` / `.venv` / `corpus` but **keeps** `chroma_index/`. Next upload was 44.2 MiB.

### 3. NVIDIA embeddings all 410 Gone

```
The model 'nvidia/nv-embedqa-e5-v5' has reached its end of life on 2026-08-25
```

Every other NIM embedding on this account was 410 or 404. Fix: local ONNX `BAAI/bge-small-en-v1.5` via `fastembed` (no PyTorch). Embeddings do not count toward the two-provider rule.

### 4. Gemini free tier is 20 requests/day

Cloud Run, 2026-10-14:

```
Quota exceeded ... GenerateRequestsPerDayPerProjectPerModel-FreeTier ... quotaValue: 20
```

The router already retries on NVIDIA (`dimension=fallback`). That is what kept the checkpointer demo answering after Flash was exhausted.

### 5. “What’s OpenClaw?” showed no answer

Cloud Run, 2026-10-26 20:20–20:23 UTC, same `POST /api/chat`:

```
20:20:38  supervisor on gemini failed: 503 Service Unavailable
20:20:43  supervisor routed to research (NVIDIA)
20:20:44  worker research on gemini started
20:22:44  worker research on gemini failed: TimeoutError     <- 120 s of silence
20:22:46  Tavily MCP tools/call (NVIDIA fallback)
20:23:01  supervisor on gemini failed: 400 "Requests ending with a model turn are not supported."
20:23:01  NVIDIA supervisor FINISH
```

The graph **did** finish (~146 s). The UI showed nothing because worker nodes call `ainvoke`, so token chunks often never reach `graph.astream`, and `main.py` only forwarded `AIMessageChunk`. After a 120 s Gemini hang the browser looked dead.

Fixes:

- Gemini worker timeout 25 s (NVIDIA 90 s), then fallback.
- Supervisor messages are `[System, Human(query)]` plus optional findings — never a trailing AI turn (stops the Gemini 400).
- Research prompt: live/current names (OpenClaw) → `tavily_search` first, not `search_docs`.
- After the graph ends, flush the last worker `AIMessage` as a `token` if nothing streamed; `done` also carries `answer`.
- Chat shows “Waiting for supervisor / Running research…” so a long turn is not a blank page.

### 6. Tavily key in Cloud Run logs

Same class of bug as Assignment 1. `httpx` logs the MCP URL at INFO, and the key was a query parameter. Fix: `https://mcp.tavily.com/mcp/` plus `Authorization: Bearer …`. Rotate the Tavily key if those old log lines are still in the project.

## Latency measurements

Warm `/health` on the live URL: **0.27–2.9 s** (2.9 s after idle). Cloud Build of the index-bearing image: **~4 min**.

| Request | Path | TTFT | Total | Notes |
|---|---|---|---|---|
| `GET /health` warm | — | — | 0.27 s | 2026-10-13 |
| `GET /health` after idle | — | — | 2.86 s | 2026-10-26 |
| Checkpointer (RAG) | research @ NVIDIA after Gemini 429 | 4.5 s | 48.7 s | 209 token events, cites `checkpointers.mdx` / `persistence.mdx` |
| “What’s OpenClaw?” (before fix) | Gemini 120 s timeout → NVIDIA + Tavily | none in UI | ~146 s | Answer existed in state; UI dropped it |
| Expected after fix (live-web) | Gemini 25 s max → NVIDIA + Tavily | ~5–30 s | ~40–80 s | Failover no longer waits 2 minutes |

Cold start (min-instances=0): instance + FastAPI + Tavily `tools/list` + first graph compile. Spec reference is ~20–25 s TTFT cold vs ~1.5–2 s warm. Our first `/health` after deploy was 0.27 s because the new revision was already serving; a true idle cold start was the 2.86 s health on Oct 26.

## Verification narrative

`GET /health` lists research / code / summary and Tavily tools (`tavily_search`, `tavily_extract`, `tavily_crawl`, `tavily_map`, `tavily_research`). A1 `/health` still lists filesystem, Tavily, and Open-Meteo.

Retrieval eval (`python -m src.eval.ragas_eval`) over the 8-question set: context precision **0.50**, context recall **0.875**. Question 1 (checkpointer) precision 0.80 / recall 1.0. Question 7 (supervisor routing) missed `workflows-agents.mdx` in top-5.

Live RAG turn cited the right pages. Live OpenClaw turn called Tavily after the Gemini timeout. pytest: 8 passed (router + ingest).

## Cost of each run

See README cost table. Short version: embeddings $0, Tavily student/free, Cloud Run on USC credit, Gemini $0 until 20/day then NVIDIA. A docs-only lookup is one supervisor call + one worker + 5 Chroma hits. A live-web question adds one Tavily `tools/call`. The broken OpenClaw turn burned one Gemini 503, one 120 s Gemini timeout, two NVIDIA chat completions, and one Tavily search — that is why it felt expensive and slow.

## What I learned

The adapter and the supervisor graph will not tell you that Cloud Build followed `.gitignore`, that Gemini 400s when the last message is an AI turn, or that `ainvoke` inside a node does not fill the SSE token stream. Those showed up in Cloud Run logs and an empty chat bubble. The two-provider router only matters if the fallback actually fires *and* the UI prints the result.
