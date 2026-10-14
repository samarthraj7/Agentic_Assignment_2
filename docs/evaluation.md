# Evaluation

Measured 2026-10-13 against the live Cloud Run URL  
`https://multi-agent-rag-881242810047.us-west1.run.app`  
in project `csci-599-agenticai`, service `multi-agent-rag`, revision `multi-agent-rag-00001-vc8`.  
Gemini 3.8 Flash free-tier generate-content quota (20 requests/day) was already exhausted, so worker generation used the documented NVIDIA fallback. That is the representative **warm** path below.

## Test question set

| # | Question | Expected source | This pass |
|---|---|---|---|
| 1 | What does a LangGraph checkpointer store? | `langgraph/checkpointers.mdx`, `persistence.mdx` | Retrieved those pages (scores ~0.78–0.86). Answer cited both. |
| 2 | How do you stream tokens from a compiled LangGraph? | `langgraph/streaming.mdx` | Not re-run (Gemini daily cap). |
| 3 | How does FastAPI bind to Cloud Run's `PORT`? | FastAPI deployment docs | Not re-run. |
| 4 | What is the difference between a LangGraph subgraph and a node? | `langgraph/use-subgraphs.mdx` | Not re-run. |
| 5 | How do you add a system prompt to `create_react_agent`? | LangGraph / LangChain agent docs | Not re-run. |
| 6 | What is `StreamingResponse` in FastAPI used for? | FastAPI streaming docs | Not re-run. |
| 7 | How does LangGraph supervisor routing typically work? | `langgraph/workflows-agents.mdx` | Not re-run. |
| 8 | What is the role of `add_messages` in a LangGraph state schema? | LangGraph graph API docs | Not re-run. |

Local ingest/retrieval (before deploy) for “What is a LangGraph checkpointer?” returned `checkpointers.mdx`, `use-time-travel.mdx`, `sql-agent.mdx`, `persistence.mdx` as the top-5, all LangGraph persistence docs.

## RAGAS

Not run. There is no `src/eval/ragas_eval.py` yet, and the Gemini free-tier cap blocks using Flash as a judge today. Retrieval quality is evidenced by citation ids and scores on question 1, not by RAGAS aggregates.

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | — | pending RAGAS script + judge quota |
| Answer relevance | — | pending |
| Context precision | — | pending |
| Context recall | — | pending |

## Latency

Warm numbers are representative. Cold start is reported separately. TTFT is **time to first answer token** from the server `done` event (`ttft_ms`), not time to the first SSE `trace` frame.

| Path | Retrieval | LLM / worker | TTFT | Total | Notes |
|---|---|---|---|---|---|
| Warm `/health` | n/a | n/a | n/a | 0.27 s | Instance already up. Three agents + Tavily inventory. |
| Warm RAG, NVIDIA fallback | 3× `k=5` Chroma queries during the research turn | research worker 17.6 s | **4.5 s** | **48.7 s** | Gemini supervisor + research 429, then NVIDIA. 209 streamed token events. Second supervisor hop FINISH. |
| Warm RAG + Tavily | — | — | — | ~230 s (aborted client) | Earlier probe; Gemini 503/429 + long NVIDIA tool loop. Not a clean representative number. |
| Cloud Build + first revision | n/a | n/a | n/a | ~4 min build, then revision rollout | `gcloud builds submit` of 44.2 MiB (includes `chroma_index`). |
| Cold start (min-instances=0) | — | — | not isolated | — | First `/health` after deploy was 0.27 s because the new revision was already serving. A true idle cold start was not measured this pass. Spec reference: ~20–25 s TTFT cold vs ~1.5–2 s warm. |

Breakdown of the measured RAG turn (server clock):

1. Supervisor Gemini 429 → NVIDIA structured route to `research` (~first 4–5 s, first token 4533 ms).
2. Research Gemini 429 → NVIDIA + three `search_docs` calls.
3. Worker done at 17612 ms; supervisor Gemini 429 → NVIDIA FINISH.
4. Citations emitted; `done.total_ms` = 48663.

## Cost per query

Pricing page date: 2026-10-13. Embeddings are local (`BAAI/bge-small-en-v1.5`) so ingest and retrieval are $0 API. Tavily student/free plan. Cloud Run on USC education credit (1 GiB, min-instances=0).

| Query class | Gemini | NVIDIA | Embeddings | Tavily | Cloud Run | Total (API) |
|---|---|---|---|---|---|---|
| Short lookup (intended) | Flash free tier, $0 until 20 req/day | unused if Gemini succeeds | $0 | $0 if docs-only | credit | **$0** on free tier |
| Measured RAG turn (this pass) | 3× 429, $0 | 1 research generation + 2 supervisor structured calls | $0 | unused | credit | NVIDIA token cost only (SDK did not expose counts on this pass) |
| Multi-step / complexity ≥ 2 | supervisor Flash | Nemotron Ultra generation | $0 | if live web | credit | NVIDIA-dominated |

Gemini paid list price if billing is enabled: $0.75 / $3.75 per 1M input/output tokens through 2026-12-31 ([Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)). This project stayed on the free tier and hit the 20-request daily cap.

## Provider comparison

| Dimension | Gemini 3.8 Flash | Nemotron 3 Ultra |
|---|---|---|
| TTFT (this pass) | failed (429 after 20 free requests; also 503 high demand) | 4.5 s to first token on the measured RAG turn |
| Answer quality | not isolated today | Grounded checkpointer answer with correct citation ids |
| Cost | $0 on free tier, then hard stop | student/free NIM; used as fallback |
| When the router picks it | supervisor, summary, complexity < 2 | complexity ≥ 2, or any Gemini failure (`dimension=fallback`) |

The trace on the measured turn showed `llm_route` → Gemini, `error` 429, `llm_route` NVIDIA fallback, `rag_retrieval`, `worker_done`, supervisor `FINISH`. Routing is inspectable in the UI; it is not a hard-coded single provider.
