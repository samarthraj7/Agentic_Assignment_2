"""Hybrid cost/capability router across two providers (Google Gemini, NVIDIA NIM).

Every decision returns a Route with a human-readable reason; the graph emits it
as an `llm_route` trace event so the UI shows which provider got the call and why.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src import config

FAST = "gemini"
CAPABLE = "nvidia"

COMPLEX_HINTS = re.compile(
    r"\b(compare|versus|vs\.?|trade-?offs?|why|design|architect\w*|step[- ]by[- ]step|"
    r"explain how|pros and cons|implement|refactor|debug|migrate|end[- ]to[- ]end)\b",
    re.I,
)


@dataclass
class Route:
    provider: str
    model: str
    dimension: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _model_id(provider: str) -> str:
    return config.GEMINI_MODEL if provider == "gemini" else config.NVIDIA_MODEL


def available(provider: str) -> bool:
    return bool(config.GOOGLE_API_KEY if provider == "gemini" else config.NVIDIA_API_KEY)


def complexity(query: str) -> int:
    score = len(COMPLEX_HINTS.findall(query))
    score += len(query.split()) > 30
    score += query.count("?") > 1
    return score


def route(role: str, query: str) -> Route:
    if role == "supervisor":
        provider, dim = FAST, "cost"
        reason = "Supervisor routing is a short classification; cheapest/fastest provider."
    elif role == "summary":
        provider, dim = FAST, "capability"
        reason = "Summarization over long retrieved context; Gemini Flash has the larger context window at lower cost."
    else:
        score = complexity(query)
        if score >= 2:
            provider, dim = CAPABLE, "capability"
            reason = f"Complexity score {score} (multi-step synthesis) -> larger reasoning model."
        else:
            provider, dim = FAST, "cost"
            reason = f"Complexity score {score} (short grounded lookup) -> cheaper fast model."

    if not available(provider):
        other = CAPABLE if provider == FAST else FAST
        return Route(other, _model_id(other), "fallback", f"{reason} But {provider} has no API key configured; using {other}.")
    return Route(provider, _model_id(provider), dim, reason)


def alternate(r: Route) -> Route | None:
    other = CAPABLE if r.provider == FAST else FAST
    if not available(other):
        return None
    return Route(other, _model_id(other), "fallback", f"{r.provider} call failed; retried on {other}.")


def build_llm(r: Route, temperature: float = 0.2) -> BaseChatModel:
    if r.provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=r.model, google_api_key=config.GOOGLE_API_KEY, temperature=temperature, max_retries=1
        )
    # NVIDIA NIM is OpenAI-compatible; thinking traces off so streamed tokens are the answer.
    return ChatOpenAI(
        model=r.model,
        api_key=config.NVIDIA_API_KEY,
        base_url=config.NVIDIA_BASE_URL,
        temperature=temperature,
        max_tokens=2048,
        max_retries=1,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
