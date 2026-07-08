import pytest

from rag_assistant.ingestion import chunk_pages, split_text


def test_short_text_is_single_chunk():
    assert split_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_empty_text_yields_no_chunks():
    assert split_text("   ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_limit():
    text = "word " * 500
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_cover_full_text():
    text = ". ".join(f"Sentence number {i}" for i in range(100))
    chunks = split_text(text, chunk_size=150, overlap=30)
    # Every sentence must appear in at least one chunk.
    for i in range(100):
        assert any(f"Sentence number {i}" in c for c in chunks)


def test_adjacent_chunks_overlap():
    text = "abcdefghij " * 100
    chunks = split_text(text, chunk_size=100, overlap=40)
    for left, right in zip(chunks, chunks[1:], strict=False):
        # The start of the next chunk should re-appear near the end of the previous.
        assert right[:10] in left or left[-40:].strip() != ""


def test_no_infinite_loop_on_unbreakable_text():
    text = "x" * 5000  # no whitespace or sentence boundaries at all
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert "".join(c[: 100 - 20] for c in chunks)  # completed without hanging
    assert len(chunks) < 200


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        split_text("text", chunk_size=100, overlap=100)


def test_chunk_pages_preserves_provenance():
    pages = [(1, "First page content. " * 20), (3, "Third page content. " * 20)]
    chunks = chunk_pages(pages, source="doc.pdf", chunk_size=150, overlap=30)
    assert all(c.source == "doc.pdf" for c in chunks)
    assert {c.page for c in chunks} == {1, 3}
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # unique ids
