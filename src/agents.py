"""The three specialized workers: distinct system prompts and distinct tool access."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, tool

from src.rag.retriever import fetch_document, search_code_examples, search_docs

CITE_RULE = (
    "Cite every claim that comes from the docs index with its citation id in square brackets, "
    "exactly as given, e.g. [langgraph/persistence.mdx#3]. Cite web results with their URL. "
    "Never invent citation ids. If the tools return nothing relevant, say so plainly."
)


@tool
def check_python(code: str) -> str:
    """Parse a Python snippet and report syntax errors, imported modules, and defined functions/classes.
    Use it to verify code before returning it."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError line {exc.lineno}: {exc.msg}"
    imports = sorted({a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
                     | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module})
    defs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return f"OK. imports={imports} defs={defs}"


@dataclass
class AgentSpec:
    name: str
    description: str
    prompt: str
    local_tools: list[BaseTool]
    mcp_tools: list[str] = field(default_factory=list)


AGENTS: dict[str, AgentSpec] = {
    "research": AgentSpec(
        name="research",
        description="Answers factual/conceptual questions about LangGraph, LangChain, FastAPI, and current events, "
                    "grounded in the docs index plus live web search.",
        prompt=(
            "You are the Research Agent. Ground every answer in retrieved evidence. "
            "Use search_docs for anything about LangGraph, LangChain, or FastAPI. "
            "Use tavily_search for current events, releases, versions, or anything newer than the docs. "
            "Prefer the docs index when it answers the question; combine both when useful. " + CITE_RULE
        ),
        local_tools=[search_docs],
        mcp_tools=["tavily_search"],
    ),
    "code": AgentSpec(
        name="code",
        description="Writes, explains, reviews, or debugs Python code using LangGraph, LangChain, or FastAPI APIs.",
        prompt=(
            "You are the Code Agent. Write or analyze Python using only APIs shown in documented examples. "
            "Always call search_code_examples first to find the documented API shape. "
            "Run check_python on any code you produce and fix errors before answering. "
            "Return code in fenced ```python blocks with a short explanation. " + CITE_RULE
        ),
        local_tools=[search_code_examples, check_python],
    ),
    "summary": AgentSpec(
        name="summary",
        description="Summarizes a documentation page, a topic across pages, a web page/URL, or other agents' findings.",
        prompt=(
            "You are the Summary Agent. Produce concise, structured summaries that preserve citations. "
            "For a docs page, find it with search_docs, then load the whole page with fetch_document. "
            "For a URL, use tavily_extract; for a recent topic, use tavily_search. "
            "If other agents already produced findings, condense them and keep their citation ids. "
            "Use bullet points; keep every citation attached to the claim it supports. " + CITE_RULE
        ),
        local_tools=[search_docs, fetch_document],
        mcp_tools=["tavily_search", "tavily_extract"],
    ),
}
