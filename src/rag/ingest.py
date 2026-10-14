"""Ingestion: load corpus -> clean MDX -> header-aware chunking -> embed -> Chroma.

Run: python -m src.rag.ingest [--reset]
Chunk IDs are deterministic (source path + chunk index), so re-running upserts
instead of duplicating; --reset drops the collection first.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src import config
from src.rag.retriever import get_store

log = logging.getLogger("ingest")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
BATCH = 64

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
JS_ONLY = re.compile(r"^:::js\n.*?^:::\n", re.S | re.M)
FENCE_MARKERS = re.compile(r"^:::(python)?\s*$", re.M)
MDX_IMPORT = re.compile(r"^(import|export) .*$", re.M)
HEADER_ANCHOR = re.compile(r"\s*\{ #[\w-]+ \}$", re.M)

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def source_url(rel: Path) -> str:
    parts = rel.with_suffix("").parts
    if parts[0] == "fastapi":
        path = "/".join(p for p in parts[1:] if p != "index")
        return f"https://fastapi.tiangolo.com/{path}/".replace("//", "/").replace(":/", "://")
    return f"https://docs.langchain.com/oss/python/{'/'.join(parts)}"


def clean(text: str) -> tuple[str, str]:
    title = ""
    m = FRONTMATTER.match(text)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip("\"'") if t else ""
        text = text[m.end():]
    text = JS_ONLY.sub("", text)
    text = FENCE_MARKERS.sub("", text)
    text = MDX_IMPORT.sub("", text)
    text = HEADER_ANCHOR.sub("", text)
    if not title:
        h1 = re.search(r"^# (.+)$", text, re.M)
        title = h1.group(1).strip() if h1 else ""
    return text.strip(), title


def load_corpus(root: Path) -> list[Document]:
    docs = []
    for path in sorted(root.rglob("*.md*")):
        if path.suffix not in {".md", ".mdx"}:
            continue
        rel = path.relative_to(root)
        body, title = clean(path.read_text(encoding="utf-8", errors="ignore"))
        if len(body) < 200:
            continue
        docs.append(Document(
            page_content=body,
            metadata={"source": str(rel), "title": title or rel.stem, "url": source_url(rel)},
        ))
    return docs


def chunk(docs: list[Document]) -> list[Document]:
    by_header = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    by_size = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc in docs:
        sections = by_header.split_text(doc.page_content)
        for section in by_size.split_documents(sections):
            section.metadata = {
                **doc.metadata,
                "section": " > ".join(section.metadata[h] for _, h in HEADERS if h in section.metadata),
                "has_code": "```" in section.page_content,
            }
            chunks.append(section)
    counters: dict[str, int] = {}
    for c in chunks:
        i = counters[c.metadata["source"]] = counters.get(c.metadata["source"], -1) + 1
        c.metadata["chunk"] = i
        c.id = hashlib.sha1(f"{c.metadata['source']}#{i}".encode()).hexdigest()[:16]
    return chunks


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop the collection before indexing")
    args = parser.parse_args()

    if not config.CORPUS_DIR.is_dir():
        log.error("corpus dir %s missing; run scripts/fetch_corpus.sh first", config.CORPUS_DIR)
        return 1
    docs = load_corpus(config.CORPUS_DIR)
    chunks = chunk(docs)
    log.info("loaded %d documents -> %d chunks", len(docs), len(chunks))

    store = get_store()
    if args.reset:
        store.reset_collection()
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        try:
            store.add_documents(batch, ids=[c.id for c in batch])
        except Exception:
            log.exception("batch starting at %d failed; re-run to resume (ids are idempotent)", start)
            return 1
        log.info("indexed %d/%d", start + len(batch), len(chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
