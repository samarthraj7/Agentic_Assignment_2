"""CSCI 599 Assignment 2 — multi-agent RAG system (FastAPI + LangGraph supervisor + SSE)."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src import config
from src.agents import AGENTS
from src.graph import build_graph
from src.mcp_tools import load_tavily_tools

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC = config.ROOT / "static"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tools, status = await load_tavily_tools()
    app.state.mcp = {"tavily": {"status": status, "tools": [t.name for t in tools]}}
    async with AsyncSqliteSaver.from_conn_string(config.SQLITE_STATE_PATH) as saver:
        app.state.graph = build_graph(tools, saver)
        yield


app = FastAPI(title="CSCI 599 Assignment 2", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "agents": list(AGENTS), "mcp": getattr(app.state, "mcp", {}), "state_backend": config.STATE_BACKEND}


def sse(event: str, data: Any) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, default=str)}


async def run_turn(body: ChatRequest) -> AsyncIterator[dict[str, str]]:
    graph = app.state.graph
    cfg = {"configurable": {"thread_id": body.session_id}, "recursion_limit": 25}
    inputs = {"messages": [HumanMessage(body.query)], "turn": [], "citations": [], "hops": 0}
    start = time.perf_counter()
    first_token: float | None = None
    try:
        async for _ns, mode, chunk in graph.astream(inputs, cfg, stream_mode=["messages", "custom"], subgraphs=True):
            if mode == "custom":
                yield sse("error" if chunk.get("type") == "error" else "trace", chunk)
                continue
            msg, meta = chunk
            tags = meta.get("tags") or []
            if not isinstance(msg, AIMessageChunk) or "supervisor" in tags or not msg.text:
                continue
            if first_token is None:
                first_token = time.perf_counter()
            agent = next((t.split(":", 1)[1] for t in tags if t.startswith("worker:")), "")
            yield sse("token", {"text": msg.text, "agent": agent})

        final = await graph.aget_state(cfg)
        seen = set()
        for c in final.values.get("citations", []):
            if c["id"] not in seen:
                seen.add(c["id"])
                yield sse("citation", c)
        yield sse("done", {
            "total_ms": round((time.perf_counter() - start) * 1000),
            "ttft_ms": round((first_token - start) * 1000) if first_token else None,
            "agents": final.values.get("turn", []),
        })
    except Exception as exc:
        log.exception("turn failed")
        yield sse("error", {"where": "graph", "detail": f"{type(exc).__name__}: {exc}"[:300]})
        yield sse("done", {"total_ms": round((time.perf_counter() - start) * 1000), "failed": True})


@app.post("/api/chat")
async def chat(body: ChatRequest) -> EventSourceResponse:
    return EventSourceResponse(run_turn(body), ping=15)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
