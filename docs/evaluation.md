# Evaluation

Numbers below are placeholders. Fill them after a verification pass against the live Cloud Run URL.

## Test question set (RAGAS)

| # | Question | Expected source |
|---|---|---|
| 1 | What does a LangGraph checkpointer store, and why would you use one? | `langgraph/persistence.mdx` / `checkpointers.mdx` |
| 2 | How do you stream tokens from a compiled LangGraph? | `langgraph/streaming.mdx` |
| 3 | How does FastAPI bind to Cloud Run's `PORT`? | FastAPI deployment docs |
| 4 | What is the difference between a LangGraph subgraph and a node? | `langgraph/use-subgraphs.mdx` |
| 5 | How do you add a system prompt to `create_react_agent`? | LangGraph / LangChain agent docs |
| 6 | What is `StreamingResponse` in FastAPI used for? | FastAPI streaming docs |
| 7 | How does LangGraph supervisor routing typically work? | `langgraph/workflows-agents.mdx` |
| 8 | What is the role of `add_messages` in a LangGraph state schema? | LangGraph graph API docs |

## RAGAS (to fill)

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | — | |
| Answer relevance | — | |
| Context precision | — | |
| Context recall | — | |

Judge model: *(record at eval time)*. Command: `python -m src.eval.ragas_eval` *(not written yet)*.

## Latency (to fill)

Report warm numbers as representative; note cold start separately.

| Path | Retrieval | LLM | TTFT | Total | Notes |
|---|---|---|---|---|---|
| Warm, RAG only | — | — | — | — | |
| Warm, RAG + Tavily | — | — | — | — | |
| Cold start (min-instances=0) | — | — | — | — | |

## Cost per query (to fill)

Record provider, exact model id, input/output tokens, and the pricing page date.

| Query class | Gemini tokens / $ | NVIDIA tokens / $ | Embeddings | Tavily | Cloud Run | Total |
|---|---|---|---|---|---|---|
| Short lookup | — | — | local / $0 | — | — | — |
| Multi-step synthesis | — | — | local / $0 | — | — | — |

## Provider comparison (to fill)

| Dimension | Gemini 3.8 Flash | Nemotron 3 Ultra |
|---|---|---|
| TTFT | — | — |
| Answer quality on the test set | — | — |
| Cost per typical query | — | — |
| When the router picks it | cheap / short / summary | complexity ≥ 2 |
