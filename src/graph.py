"""Supervisor pattern: supervisor -> worker -> supervisor ... -> END.

Adapted from https://docs.langchain.com/oss/python/langgraph/workflows-agents — supervisor
routing via structured output and conditional edges on a state field (`next`).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src import config
from src.agents import AGENTS
from src.llm_router import Route, alternate, build_llm, route
from src.mcp_tools import for_agent

log = logging.getLogger("graph")

WORKER_TIMEOUT_S = {"gemini": 25, "nvidia": 90}
HISTORY = 6


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next: str
    reason: str
    hops: int
    turn: list[dict]
    citations: list[dict]


class Decision(BaseModel):
    next: Literal["research", "code", "summary", "FINISH"] = Field(description="worker to run next, or FINISH")
    reason: str = Field(description="one sentence: why this worker (or why done)")


SUPERVISOR_PROMPT = """You are the supervisor of a multi-agent system. Workers:
{workers}

Pick the single next worker for the user's latest request, or FINISH once the workers' output fully answers it.
- Pick a worker first; never FINISH before any worker has run this turn.
- Chain workers only when the request needs it (e.g. research then summary, or research then code).
- Do not re-run a worker that already succeeded this turn.
Workers already run this turn: {done}"""

CODE_HINT = re.compile(r"\b(code|function|class|implement|snippet|bug|debug|script|write .* (app|endpoint|graph))\b", re.I)
SUMMARY_HINT = re.compile(r"\b(summari[sz]e|summary|tl;?dr|overview of|condense)\b", re.I)


def latest_query(state: State) -> str:
    return next((m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")


def keyword_fallback(query: str, done: list[str]) -> Decision:
    if done:
        return Decision(next="FINISH", reason="Supervisor LLM unavailable; finishing after first worker.")
    pick = "code" if CODE_HINT.search(query) else "summary" if SUMMARY_HINT.search(query) else "research"
    return Decision(next=pick, reason=f"Supervisor LLM unavailable; keyword fallback chose {pick}.")


async def supervisor(state: State) -> dict[str, Any]:
    writer = get_stream_writer()
    query = latest_query(state)
    done = [t["agent"] for t in state.get("turn", []) if t["ok"]]
    hops = state.get("hops", 0)

    if hops >= config.MAX_SUPERVISOR_HOPS:
        decision = Decision(next="FINISH", reason=f"Hop limit {config.MAX_SUPERVISOR_HOPS} reached.")
    else:
        r = route("supervisor", query)
        writer({"type": "llm_route", "agent": "supervisor", **r.as_dict()})
        workers = "\n".join(f"- {a.name}: {a.description}" for a in AGENTS.values())
        prompt = SUPERVISOR_PROMPT.format(workers=workers, done=", ".join(done) or "none")
        # Gemini 400s on "requests ending with a model turn" if the last message is AI.
        # Supervisor only needs the user query plus any worker findings from this turn.
        msgs: list = [SystemMessage(prompt), HumanMessage(query)]
        if done:
            findings = "\n\n".join(
                f"[{m.name}] {m.text[:800]}"
                for m in state["messages"]
                if isinstance(m, AIMessage) and m.name in done
            )
            if findings:
                msgs.append(HumanMessage(f"Worker findings so far:\n{findings}"))
        decision = None
        for attempt in (r, alternate(r)):
            if attempt is None:
                continue
            try:
                llm = build_llm(attempt, temperature=0).with_structured_output(Decision)
                decision = await llm.ainvoke(msgs, config={"tags": ["supervisor"]})
                break
            except Exception as exc:
                log.warning("supervisor on %s failed: %s", attempt.provider, exc)
                writer({"type": "error", "where": "supervisor", "provider": attempt.provider, "detail": type(exc).__name__})
        if decision is None:
            decision = keyword_fallback(query, done)
        if decision.next == "FINISH" and not done:
            decision = keyword_fallback(query, done)
        if decision.next in done:
            decision = Decision(next="FINISH", reason=f"{decision.next} already answered this turn.")

    writer({"type": "agent_route", "from": "supervisor", "to": decision.next, "reason": decision.reason, "hop": hops})
    return {"next": decision.next, "reason": decision.reason, "hops": hops + 1}


def make_worker(name: str, mcp_tools: dict[str, BaseTool]):
    spec = AGENTS[name]
    tools = [*spec.local_tools, *(for_agent(mcp_tools[t], name) for t in spec.mcp_tools if t in mcp_tools)]

    async def run(r: Route, state: State) -> list[AnyMessage]:
        query = latest_query(state)
        prior = [m for m in state["messages"] if isinstance(m, AIMessage) and m.name in AGENTS]
        prior_this_turn = [m for m in prior if m.id in {t.get("msg_id") for t in state.get("turn", [])}]
        context = "\n\n".join(f"[{m.name} agent findings]\n{m.text}" for m in prior_this_turn)
        history = [m for m in state["messages"][-HISTORY:] if m not in prior_this_turn][:-1]
        task = HumanMessage(f"{query}\n\n{context}" if context else query)
        agent = create_react_agent(build_llm(r), tools, prompt=spec.prompt, name=name, checkpointer=False)
        out = await asyncio.wait_for(
            agent.ainvoke({"messages": [*history, task]}, config={"recursion_limit": 8, "tags": [f"worker:{name}"]}),
            WORKER_TIMEOUT_S.get(r.provider, 90),
        )
        return out["messages"]

    async def node(state: State) -> dict[str, Any]:
        writer = get_stream_writer()
        start = time.perf_counter()
        r = route(name, latest_query(state))
        writer({"type": "llm_route", "agent": name, **r.as_dict()})
        messages, error = None, None
        for attempt in (r, alternate(r)):
            if attempt is None:
                continue
            if attempt is not r:
                writer({"type": "llm_route", "agent": name, **attempt.as_dict()})
            try:
                messages = await run(attempt, state)
                r = attempt
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("worker %s on %s failed: %s", name, attempt.provider, error)
                writer({"type": "error", "where": name, "provider": attempt.provider, "detail": error})

        ms = round((time.perf_counter() - start) * 1000)
        if messages is None:
            msg = AIMessage(f"The {name} agent failed ({error}). Answering without it.", name=name, id=str(uuid4()))
            writer({"type": "worker_done", "agent": name, "ok": False, "ms": ms})
            return {"messages": [msg], "turn": [*state.get("turn", []), {"agent": name, "ok": False, "msg_id": msg.id}]}

        citations = [c for m in messages for c in (getattr(m, "artifact", None) or []) if isinstance(c, dict)]
        final = AIMessage(messages[-1].text, name=name, id=str(uuid4()))
        writer({"type": "worker_done", "agent": name, "ok": True, "ms": ms, "provider": r.provider,
                "tool_calls": sum(len(getattr(m, "tool_calls", []) or []) for m in messages)})
        return {
            "messages": [final],
            "turn": [*state.get("turn", []), {"agent": name, "ok": True, "provider": r.provider, "msg_id": final.id}],
            "citations": [*state.get("citations", []), *citations],
        }

    return node


def build_graph(mcp_tools: list[BaseTool], checkpointer):
    by_name = {t.name: t for t in mcp_tools}
    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    for name in AGENTS:
        g.add_node(name, make_worker(name, by_name))
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"], {**{n: n for n in AGENTS}, "FINISH": END})
    return g.compile(checkpointer=checkpointer)
