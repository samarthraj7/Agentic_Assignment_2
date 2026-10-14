from src.llm_router import complexity, route


def test_supervisor_is_cheap():
    r = route("supervisor", "anything")
    assert r.dimension in {"cost", "fallback"}
    assert r.reason


def test_summary_is_capability_or_fallback():
    r = route("summary", "Summarize persistence.mdx")
    assert r.dimension in {"capability", "fallback"}


def test_complex_query_scores_high():
    q = "Compare LangGraph checkpoints versus subgraphs and explain how to implement a supervisor."
    assert complexity(q) >= 2
    r = route("research", q)
    assert r.dimension in {"capability", "fallback"}


def test_simple_query_scores_low():
    assert complexity("What is a checkpointer?") < 2
