import pytest

from conftest import FakeAnthropicClient
from rag_assistant.llm import AnswerService, MissingAPIKeyError, format_sources
from rag_assistant.vectorstore import RetrievedChunk

CHUNKS = [
    RetrievedChunk(
        chunk_id="a", text="Aspirin dose is 81 mg daily.", source="guide.pdf", page=4, score=0.9
    ),
    RetrievedChunk(
        chunk_id="b", text="Store below 25 degrees.", source="guide.pdf", page=9, score=0.5
    ),
]


def test_format_sources_numbers_and_provenance():
    formatted = format_sources(CHUNKS)
    assert "[1] (guide.pdf, page 4)" in formatted
    assert "[2] (guide.pdf, page 9)" in formatted


def test_answer_resolves_citations_to_pages():
    fake = FakeAnthropicClient(
        {"answer": "The dose is 81 mg daily.", "citations": [{"source": 1, "quote": "81 mg daily"}]}
    )
    service = AnswerService(client=fake, model="test-model")
    result = service.answer("What is the aspirin dose?", CHUNKS)

    assert result.grounded
    assert result.answer == "The dose is 81 mg daily."
    assert result.citations[0].source == "guide.pdf"
    assert result.citations[0].page == 4

    # Request must be schema-constrained and grounded on the provided sources only.
    call = fake.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "81 mg" in call["messages"][0]["content"]


def test_no_citations_marks_ungrounded():
    fake = FakeAnthropicClient({"answer": "The documents do not contain this.", "citations": []})
    result = AnswerService(client=fake, model="test-model").answer("Capital of Mongolia?", CHUNKS)
    assert not result.grounded
    assert result.citations == []


def test_empty_retrieval_short_circuits_without_api_call():
    fake = FakeAnthropicClient({"answer": "unused", "citations": []})
    result = AnswerService(client=fake, model="test-model").answer("anything", [])
    assert not result.grounded
    assert fake.calls == []


def test_refusal_stop_reason_handled():
    fake = FakeAnthropicClient("", stop_reason="refusal")
    result = AnswerService(client=fake, model="test-model").answer("q", CHUNKS)
    assert not result.grounded


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = AnswerService(model="test-model")
    service._api_key = None  # simulate no key in .env either
    with pytest.raises(MissingAPIKeyError, match="console.anthropic.com"):
        _ = service.client


def test_placeholder_api_key_rejected(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = AnswerService(model="test-model")
    service._api_key = "sk-ant-..."  # the untouched .env.example placeholder
    with pytest.raises(MissingAPIKeyError):
        _ = service.client


def test_out_of_range_citation_index_does_not_crash():
    fake = FakeAnthropicClient(
        {"answer": "ok", "citations": [{"source": 99, "quote": "ghost"}]}
    )
    result = AnswerService(client=fake, model="test-model").answer("q", CHUNKS)
    assert result.citations[0].source == ""
    assert result.citations[0].page == 0
