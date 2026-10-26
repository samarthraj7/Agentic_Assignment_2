# Evaluation

Measured against `https://multi-agent-rag-881242810047.us-west1.run.app`  
project `csci-599-agenticai`, service `multi-agent-rag`.  
Retrieval metrics: `python -m src.eval.ragas_eval` on 2026-10-26 (local Chroma, same index as the image).

## Test question set

| # | Question | Expected source | Retrieval this pass |
|---|---|---|---|
| 1 | What does a LangGraph checkpointer store, and why would you use one? | `checkpointers.mdx`, `persistence.mdx` | precision 0.80, recall 1.0. Live answer cited both. |
| 2 | How do you stream tokens from a compiled LangGraph? | `streaming.mdx` | precision 0.80, recall 1.0 |
| 3 | How does FastAPI bind to Cloud Run's `PORT`? | FastAPI deployment docs | precision 0.20, recall 1.0 (deployment/concepts.md in top-5) |
| 4 | What is the difference between a LangGraph subgraph and a node? | `use-subgraphs.mdx` | precision 0.20, recall 1.0 |
| 5 | How do you add a system prompt to `create_react_agent`? | LangChain / LangGraph agent docs | precision 0.80, recall 1.0 |
| 6 | What is `StreamingResponse` in FastAPI used for? | FastAPI streaming / custom-response | precision 0.20, recall 1.0 (`stream-data.md`, `custom-response.md`) |
| 7 | How does LangGraph supervisor routing typically work? | `workflows-agents.mdx` | precision 0.00, recall 0.0 (landed on multi-agent/subagents pages) |
| 8 | What is the role of `add_messages` in a LangGraph state schema? | `graph-api.mdx` | precision 1.00, recall 1.0 |

## RAGAS-style metrics

Full RAGAS LLM-as-judge (faithfulness, answer relevance) needs a generation judge. Gemini Flash free tier is 20 requests/day and was exhausted during live verification, so those two scores are not invented. The two **retrieval** RAGAS metrics are computed over the fixed 8-question set as context precision / context recall against expected source paths (`src/eval/ragas_eval.py`).

| Metric | Score | How |
|---|---|---|
| Context precision | **0.50** | relevant hits / k, averaged over 8 questions |
| Context recall | **0.875** | 7/8 questions had at least one expected source in top-5 |
| Faithfulness | — | not judged; live checkpointer answer stayed inside retrieved chunks |
| Answer relevance | — | not judged; live checkpointer answer addressed the question |

Question 7 is the miss: supervisor routing docs live under LangChain multi-agent pages in this corpus snapshot, not `workflows-agents.mdx`.

## Latency

TTFT is time to first **answer** token (`done.ttft_ms`). Warm numbers are representative.

| Path | Retrieval | Worker | TTFT | Total | Notes |
|---|---|---|---|---|---|
| Warm `/health` | — | — | — | 0.27 s | 2026-10-13 |
| `/health` after idle | — | — | — | 2.86 s | 2026-10-26 |
| Warm RAG, NVIDIA fallback | 3× k=5 | 17.6 s | **4.5 s** | **48.7 s** | Gemini 429 then NVIDIA. Citations on checkpointer pages. |
| OpenClaw live-web (before fix) | Tavily after 120 s Gemini timeout | NVIDIA | none in UI | **~146 s** | Cloud Run 20:20–20:23 UTC. Answer in state, UI blank. |
| Cloud Build + revision | — | — | — | ~4 min | 44.2 MiB including `chroma_index` |

Cold start: min-instances=0. Spec reference ~20–25 s TTFT cold vs ~1.5–2 s warm. We did not isolate a chat cold-start TTFT; idle health was 2.86 s.

## Cost per query

See README “Cost of each run”. Embeddings $0. Tavily student/free. Cloud Run on USC credit. Gemini $0 until 20 req/day, then NVIDIA. The OpenClaw failure mode (Gemini 503 + 120 s timeout + NVIDIA + Tavily) is the expensive shape we fixed by failing Gemini at 25 s.

## Provider comparison

| Dimension | Gemini 3.8 Flash | Nemotron 3 Ultra |
|---|---|---|
| When routed | supervisor, summary, complexity < 2 | complexity ≥ 2, or any Gemini failure |
| This project | 20 free req/day; 429 / 503 / 120 s hang observed | Produced the measured RAG answer and the OpenClaw Tavily turn |
| TTFT (measured RAG) | n/a (429) | 4.5 s |
| Cost | $0 on free tier, then hard stop | student/free NIM |

Routing is inspectable: `llm_route` + `error` + `llm_route dimension=fallback` in the UI.
