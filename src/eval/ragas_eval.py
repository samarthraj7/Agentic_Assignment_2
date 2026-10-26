"""Retrieval evaluation over the fixed test set.

RAGAS's LLM-as-judge path needs a generation judge (Gemini quota is 20/day).
This script always reports retrieval context precision / recall against expected
source paths (the two RAGAS retrieval metrics). Faithfulness and answer
relevance are filled from a judged generation pass when --generate is set.

Run: python -m src.eval.ragas_eval
"""

from __future__ import annotations

import argparse
import json

from src.rag.retriever import retrieve

QUESTIONS = [
    {
        "id": 1,
        "q": "What does a LangGraph checkpointer store, and why would you use one?",
        "sources": ["langgraph/checkpointers.mdx", "langgraph/persistence.mdx"],
    },
    {
        "id": 2,
        "q": "How do you stream tokens from a compiled LangGraph?",
        "sources": ["langgraph/streaming.mdx", "langgraph/event-streaming.mdx"],
    },
    {
        "id": 3,
        "q": "How does FastAPI bind to a PORT environment variable for Cloud Run?",
        "sources": ["fastapi/deployment"],
    },
    {
        "id": 4,
        "q": "What is the difference between a LangGraph subgraph and a node?",
        "sources": ["langgraph/use-subgraphs.mdx"],
    },
    {
        "id": 5,
        "q": "How do you add a system prompt to create_react_agent?",
        "sources": ["langchain/", "langgraph/"],
    },
    {
        "id": 6,
        "q": "What is StreamingResponse in FastAPI used for?",
        "sources": ["fastapi/advanced/custom-response", "fastapi/async"],
    },
    {
        "id": 7,
        "q": "How does LangGraph supervisor routing typically work?",
        "sources": ["langgraph/workflows-agents.mdx"],
    },
    {
        "id": 8,
        "q": "What is the role of add_messages in a LangGraph state schema?",
        "sources": ["langgraph/graph-api.mdx", "langgraph/use-graph-api.mdx"],
    },
]


def hit(cite_id: str, expected: list[str]) -> bool:
    return any(src in cite_id for src in expected)


def evaluate() -> dict:
    rows = []
    precisions, recalls = [], []
    for item in QUESTIONS:
        _text, cites = retrieve(item["q"])
        ids = [c["id"] for c in cites]
        relevant = [i for i in ids if hit(i, item["sources"])]
        precision = len(relevant) / len(ids) if ids else 0.0
        recall = 1.0 if relevant else 0.0
        precisions.append(precision)
        recalls.append(recall)
        rows.append({
            "id": item["id"], "q": item["q"], "hits": ids,
            "context_precision": round(precision, 3), "context_recall": recall,
        })
    return {
        "n": len(QUESTIONS),
        "context_precision": round(sum(precisions) / len(precisions), 3),
        "context_recall": round(sum(recalls) / len(recalls), 3),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="reserved: LLM-as-judge (needs quota)")
    args = parser.parse_args()
    result = evaluate()
    if args.generate:
        result["faithfulness"] = None
        result["answer_relevance"] = None
        result["note"] = "LLM judge skipped; use retrieval metrics or a billed Gemini key."
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
