from rag_assistant.ingestion import Chunk


def make_chunk(chunk_id: str, text: str, source: str = "doc.pdf", page: int = 1) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source=source, page=page)


def test_empty_store_returns_nothing(store):
    assert store.search("anything", k=5) == []
    assert store.count() == 0


def test_add_and_retrieve_relevant_chunk(store):
    store.add(
        [
            make_chunk("a", "The mitochondria is the powerhouse of the cell", page=2),
            make_chunk("b", "Contract termination requires ninety days written notice", page=7),
            make_chunk("c", "The stock market closed higher on Tuesday", page=9),
        ]
    )
    results = store.search("what is the powerhouse of the cell", k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "a"
    assert results[0].page == 2
    assert 0.0 <= results[0].score <= 1.0


def test_upsert_is_idempotent(store):
    chunk = make_chunk("a", "same text")
    store.add([chunk])
    store.add([chunk])
    assert store.count() == 1


def test_list_and_delete_sources(store):
    store.add(
        [
            make_chunk("a", "alpha text", source="one.pdf"),
            make_chunk("b", "beta text", source="one.pdf"),
            make_chunk("c", "gamma text", source="two.pdf"),
        ]
    )
    assert store.list_sources() == {"one.pdf": 2, "two.pdf": 1}

    store.delete_source("one.pdf")
    assert store.list_sources() == {"two.pdf": 1}
    assert store.count() == 1


def test_search_k_capped_at_count(store):
    store.add([make_chunk("a", "only one chunk here")])
    results = store.search("chunk", k=10)
    assert len(results) == 1
