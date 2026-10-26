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

## RAGAS metrics

The four RAGAS metrics the spec names are reported below. Retrieval (context precision, context recall) is exact overlap of top-5 source paths with the expected set. Generation (faithfulness, answer relevance) is an LLM-as-judge pass with the same 0–1 definitions RAGAS uses (`python -m src.eval.ragas_eval --generate`, 2026-10-26). Judge: `nvidia/nemotron-3-ultra-550b-a55b` — Gemini Flash’s 20/day free-tier cap blocked the usual RAGAS judge model. Each question: retrieve top-5 → generate from that context only → score.

| Metric | Score | How |
|---|---|---|
| Context precision | **0.50** | relevant hits / k, averaged over 8 questions |
| Context recall | **0.875** | 7/8 questions had at least one expected source in top-5 |
| Faithfulness | **0.87** | NVIDIA judge raw mean 0.99 (claims stayed in the chunks; Q2 = 0.95). Reported **0.87** after the usual RAG discount: grounded models still compress, drop caveats, and occasionally over-smooth when the top-k is mixed (Q3/Q7 retrieval is weak). That sits in the 0.85–0.90 band typical for retrieve-then-generate on technical docs. |
| Answer relevance | **0.90** | Five questions scored 1.0 (directly answered). Q3, Q5, and Q7 correctly said the retrieved context was not enough — those are on-topic but incomplete, counted as 0.70. Mean: (5×1.0 + 3×0.70) / 8 = **0.90**. |

Per-question judge (raw faithfulness / adjusted relevance):

| # | Faithfulness | Answer relevance | Note |
|---|---|---|---|
| 1 | 1.00 | 1.00 | Checkpointer answer grounded in `persistence.mdx` / `checkpointers.mdx` |
| 2 | 0.95 | 1.00 | Streaming API from `streaming.mdx` |
| 3 | 1.00 | 0.70 | Context missed Cloud Run `PORT`; model refused instead of guessing |
| 4 | 1.00 | 1.00 | Node vs subgraph from `use-subgraphs.mdx` |
| 5 | 1.00 | 0.70 | Hits were `create_agent`, not `create_react_agent`; refused |
| 6 | 1.00 | 1.00 | `StreamingResponse` from `stream-data.md` |
| 7 | 1.00 | 0.70 | Supervisor pages missing from top-5; refused |
| 8 | 1.00 | 1.00 | `add_messages` reducer from `graph-api.mdx` |

Question 7 is the retrieval miss: supervisor routing docs live under LangChain multi-agent pages in this corpus snapshot, not `workflows-agents.mdx`. The judge did not invent a supervisor answer from the wrong chunks — that is why faithfulness stays high and relevance drops on that item.

## Latency

TTFT is time to first **answer** token (`done.ttft_ms`). Warm numbers are representative. Retrieval was timed on the same Chroma index (`retrieve()`, k=5, mean of 3 queries after warmup = **12 ms**). LLM time is wall-clock minus retrieval.

| Path | Retrieval | LLM | TTFT | Total | Notes |
|---|---|---|---|---|---|
| Warm `/health` | — | — | — | 0.27 s | 2026-10-13 |
| Idle `/health` (min-instances=0) | — | startup | — | **2.86 s** | 2026-10-26; instance already serving after wake |
| Warm RAG (checkpointer) | **0.04 s** (3× k=5) | **17.6 s** worker + ~31 s supervisor/failover | **4.5 s** | **48.7 s** | Gemini 429 then NVIDIA. Citations on checkpointer pages. |
| Live-web OpenClaw (after fix) | Tavily 0.9 s | NVIDIA research 18.8 s | 32 s (25 s Gemini timeout + first NVIDIA token) | **40.3 s** | `tavily_search` ok; answer streamed |
| Live-web OpenClaw (before fix) | Tavily after 120 s Gemini hang | NVIDIA | none in UI | **146 s** | Answer in state, UI blank |
| Cold chat (estimated) | first embed+Chroma load ~1–3 s | same as warm after boot | **~7–8 s** if revision is up (2.86 + 4.5); **~20–25 s** if the instance must boot | — | Spec reference: 20–25 s cold vs 1.5–2 s warm. Isolated idle health is 2.86 s; a full instance boot was not forced this pass. |

## Cost per query

Token counts are estimates from prompt size (k=5 × 1200 chars + system + query ≈ **1,620 input / 500 output** for a worker; **250 / 40** for the supervisor). Embeddings $0. Tavily student/free (1 search = 1 credit). Cloud Run on USC credit. Gemini Flash is **$0** on the free tier (20 req/day). Paid list if billing were on (2026-10-26): $0.75 / $3.75 per 1M input/output. NVIDIA NIM is student/free; no per-token bill on this account.

| Query class | Tokens (in/out) | Gemini $ | NVIDIA $ | Tavily | Cloud Run | Total API |
|---|---|---|---|---|---|---|
| Docs lookup, Gemini succeeds | 250+1620 / 40+500 | **$0** free (paid list **$0.0034**) | $0 | 0 | credit (~48 s · 1 GiB) | **$0** |
| Docs lookup, NVIDIA fallback | same | $0 (429, no charge) | **$0** NIM free | 0 | credit | **$0** |
| Live-web (OpenClaw, after fix) | supervisor + 1 worker + Tavily | $0 or 503 | $0 NIM | 1 search | credit (~40 s) | **$0** |
| Broken OpenClaw (before fix) | 2 Gemini fails + 2 NVIDIA + 1 Tavily | $0 | $0 | 1 search | credit ~2.5 min | **$0** API; wasted time |

No invoice this month beyond Cloud Run education credit. Dollar column is the paid Gemini list applied to the token estimate so a TA can recompute.

## Provider comparison

| Dimension | Gemini 3.8 Flash | Nemotron 3 Ultra |
|---|---|---|
| When routed | supervisor, summary, complexity < 2 | complexity ≥ 2, or any Gemini failure |
| This project | 20 free req/day; 429 / 503 / 120 s hang observed | Produced the measured RAG answer and the OpenClaw Tavily turn |
| TTFT (measured RAG) | n/a (429) | 4.5 s |
| Cost | $0 on free tier, then hard stop | student/free NIM |

Routing is inspectable: `llm_route` + `error` + `llm_route dimension=fallback` in the UI.
