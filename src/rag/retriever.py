"""Retrieval tools over the Chroma index. Each returns (prompt text, citation artifacts)."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from src import config

log = logging.getLogger("rag")


class LocalEmbeddings(Embeddings):
    """fastembed (ONNX, no torch) behind the LangChain Embeddings interface."""

    def __init__(self, model: str) -> None:
        self.model = TextEmbedding(model, cache_dir=config.EMBEDDING_CACHE_DIR)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed([text]))).tolist()


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=LocalEmbeddings(config.EMBEDDING_MODEL),
        persist_directory=str(config.CHROMA_PERSIST_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _emit(event: dict) -> None:
    try:
        get_stream_writer()(event)
    except RuntimeError:
        pass  # called outside a graph run (tests, eval)


def format_hits(hits: list[tuple[Document, float]]) -> tuple[str, list[dict]]:
    blocks, citations = [], []
    for doc, score in hits:
        m = doc.metadata
        cid = f"{m['source']}#{m['chunk']}"
        citations.append({
            "id": cid, "title": m.get("title", ""), "section": m.get("section", ""),
            "url": m.get("url", ""), "score": round(1 - score, 3), "snippet": doc.page_content[:240],
        })
        blocks.append(f"[{cid}] {m.get('title', '')} — {m.get('section', '')}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks), citations


def retrieve(query: str, k: int = config.RETRIEVAL_K, code_only: bool = False) -> tuple[str, list[dict]]:
    flt = {"has_code": True} if code_only else None
    try:
        hits = get_store().similarity_search_with_score(query, k=k, filter=flt)
    except Exception as exc:
        log.exception("retrieval failed")
        return f"Retrieval failed ({type(exc).__name__}). Answer without the docs index and say so.", []
    text, citations = format_hits(hits)
    _emit({"type": "rag_retrieval", "query": query, "k": k, "code_only": code_only,
           "hits": [{"id": c["id"], "score": c["score"]} for c in citations]})
    return text or "No matching chunks in the docs index.", citations


@tool(response_format="content_and_artifact")
def search_docs(query: str) -> tuple[str, list[dict]]:
    """Semantic search over the LangGraph, LangChain, and FastAPI documentation index.
    Returns the top chunks, each tagged with a citation id like [langgraph/persistence.mdx#3]."""
    return retrieve(query)


@tool(response_format="content_and_artifact")
def search_code_examples(query: str) -> tuple[str, list[dict]]:
    """Search only documentation chunks that contain code blocks (LangGraph, LangChain, FastAPI).
    Use this to ground generated or reviewed code in documented APIs."""
    return retrieve(query, code_only=True)


@tool(response_format="content_and_artifact")
def fetch_document(source: str) -> tuple[str, list[dict]]:
    """Fetch every chunk of one documentation page, in order, given its source path
    (the part of a citation id before '#', e.g. 'langgraph/persistence.mdx')."""
    try:
        got = get_store().get(where={"source": source}, include=["documents", "metadatas"])
    except Exception as exc:
        log.exception("fetch_document failed")
        return f"Could not load {source} ({type(exc).__name__}).", []
    pairs = sorted(zip(got["documents"], got["metadatas"]), key=lambda p: p[1]["chunk"])[:40]
    if not pairs:
        return f"No document with source '{source}'. Use a source path from a citation id.", []
    text, citations = format_hits([(Document(page_content=d, metadata=m), 0.0) for d, m in pairs])
    _emit({"type": "rag_retrieval", "query": f"fetch {source}", "k": len(pairs), "hits": [{"id": c["id"]} for c in citations]})
    return text, citations
