import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from conftest import DeterministicEmbedding, FakeAnthropicClient
from rag_assistant.api import app, get_pipeline
from rag_assistant.config import Settings
from rag_assistant.llm import AnswerService
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.vectorstore import VectorStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(
        chroma_dir=str(tmp_path / "chroma"),
        upload_dir=str(tmp_path / "uploads"),
        anthropic_model="test-model",
    )
    monkeypatch.setattr("rag_assistant.api.get_settings", lambda: settings)

    fake_llm = FakeAnthropicClient(
        {"answer": "A grounded answer.", "citations": [{"source": 1, "quote": "supporting text"}]}
    )
    pipeline = RAGPipeline(
        store=VectorStore(settings.chroma_dir, embedding_function=DeterministicEmbedding()),
        answerer=AnswerService(client=fake_llm, model="test-model"),
        settings=settings,
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def blank_pdf_bytes() -> bytes:
    """A structurally valid PDF with one blank (textless) page."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_rejects_non_pdf(client):
    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_upload_rejects_textless_pdf(client):
    response = client.post(
        "/documents", files={"file": ("blank.pdf", blank_pdf_bytes(), "application/pdf")}
    )
    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]


def test_upload_rejects_corrupt_pdf(client):
    response = client.post(
        "/documents", files={"file": ("bad.pdf", b"not a real pdf", "application/pdf")}
    )
    assert response.status_code == 422


def test_documents_empty_initially(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": {}, "total_chunks": 0}


def test_delete_missing_document_404(client):
    assert client.delete("/documents/nope.pdf").status_code == 404


def test_ask_returns_grounded_answer_with_citations(client):
    # Seed the store directly (bypasses PDF parsing).
    from rag_assistant.ingestion import Chunk

    pipeline = app.dependency_overrides[get_pipeline]()
    pipeline.store.add(
        [Chunk(chunk_id="x", text="supporting text about the topic", source="doc.pdf", page=3)]
    )

    response = client.post("/ask", json={"question": "What about the topic?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["answer"] == "A grounded answer."
    assert body["citations"][0]["source"] == "doc.pdf"
    assert body["citations"][0]["page"] == 3
    assert body["retrieved"][0]["source"] == "doc.pdf"


def test_ask_validates_question_length(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
