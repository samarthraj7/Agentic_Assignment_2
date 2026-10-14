from pathlib import Path

from src.rag.ingest import chunk, clean, load_corpus, source_url


def test_clean_strips_frontmatter_and_anchors():
    raw = "---\ntitle: Hello\n---\n# Hello { #hello }\n\nBody text here.\n"
    body, title = clean(raw)
    assert title == "Hello"
    assert "{ #" not in body
    assert "Body text" in body


def test_source_url_fastapi():
    assert source_url(Path("fastapi/deployment/index.md")) == "https://fastapi.tiangolo.com/deployment/"


def test_source_url_langgraph():
    assert "langgraph/persistence" in source_url(Path("langgraph/persistence.mdx"))


def test_chunk_ids_are_stable(tmp_path):
    src = tmp_path / "langgraph"
    src.mkdir()
    (src / "one.mdx").write_text("---\ntitle: One\n---\n# One\n\n" + ("word " * 400) + "\n")
    a = {c.id: c.page_content for c in chunk(load_corpus(tmp_path))}
    b = {c.id: c.page_content for c in chunk(load_corpus(tmp_path))}
    assert a == b
    assert a
