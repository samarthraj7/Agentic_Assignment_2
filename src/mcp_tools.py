"""Tavily Remote MCP (Streamable HTTP), shared by the Research and Summary agents.

Adapted from https://github.com/langchain-ai/langchain-mcp-adapters — MultiServerMCPClient
discovers tools via MCP tools/list; calls go out as tools/call.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_stream_writer

from src import config

log = logging.getLogger("mcp")


async def load_tavily_tools() -> tuple[list[BaseTool], str]:
    """Returns (tools, status). A dead MCP endpoint yields no tools, never an exception."""
    if not config.TAVILY_API_KEY:
        return [], "TAVILY_API_KEY not set"
    # Key in Authorization, not the URL — Cloud Run httpx logs the URL at INFO.
    client = MultiServerMCPClient({
        "tavily": {
            "transport": "streamable_http",
            "url": "https://mcp.tavily.com/mcp/",
            "headers": {"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
        },
    })
    try:
        tools = await client.get_tools(server_name="tavily")
    except Exception as exc:
        log.exception("Tavily MCP tools/list failed")
        return [], f"tools/list failed: {type(exc).__name__}"
    log.info("Tavily MCP tools/list -> %s", [t.name for t in tools])
    return tools, "ok"


def for_agent(tool: BaseTool, agent: str) -> BaseTool:
    """Per-agent wrapper over a shared MCP tool: traces the caller, turns failures into text."""

    async def _run(**kwargs: Any) -> str:
        writer = get_stream_writer()
        start = time.perf_counter()
        try:
            result = await tool.ainvoke(kwargs)
            ok, out = True, result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as exc:
            log.exception("MCP tools/call %s failed", tool.name)
            ok, out = False, f"Tool '{tool.name}' failed ({type(exc).__name__}): {exc}. Continue without it."
        writer({"type": "mcp_tool_call", "server": "tavily", "tool": tool.name, "agent": agent,
                "args": kwargs, "ok": ok, "ms": round((time.perf_counter() - start) * 1000)})
        return out[:12000]

    return StructuredTool.from_function(
        coroutine=_run, name=tool.name, description=tool.description or tool.name, args_schema=tool.args_schema,
    )
