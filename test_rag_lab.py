from rag_lab import citation_ids, markdown_sections, safe_fts_query, split_words


def test_chunk_overlap():
    result = split_words("one two three four five", size=3, overlap=1)
    assert result == ["one two three", "three four five"]


def test_markdown_headings():
    assert markdown_sections("# Warranty\nTwo years.\n## Exceptions\nWear not covered.") == [
        ("Warranty", "Two years."),
        ("Exceptions", "Wear not covered."),
    ]


def test_fts_input_is_quoted():
    assert safe_fts_query('warranty OR refund:') == '"warranty" OR "OR" OR "refund"'


def test_citation_validation():
    assert citation_ids("See [S2], [S8], and [S1]", 3) == [1, 2]


def test_sqlite_index_and_search(tmp_path, monkeypatch):
    import numpy as np
    import rag_lab

    class FakeEmbeddings:
        def encode(self, texts, **kwargs):
            # Deterministic fake vectors let us test storage and FTS without downloads.
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    monkeypatch.setattr(rag_lab, "embedding_model", lambda: FakeEmbeddings())
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "policy.md").write_text("# Warranty\nStandard warranty is two years.", encoding="utf-8")
    database = tmp_path / "test.sqlite3"

    assert rag_lab.index_documents(knowledge, database) == 1
    assert rag_lab.search("warranty", db_path=database)[0]["source"] == "policy.md"

    monkeypatch.setitem(rag_lab.CHAT_PROVIDERS, rag_lab.LLM_PROVIDER, lambda system, user: "Two years [S1].")
    result = rag_lab.ask("How long is the warranty?", db_path=database)
    assert result["status"] == "answered"
    assert result["sources"][0]["source"] == "policy.md"
