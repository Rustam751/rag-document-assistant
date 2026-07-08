"""End-to-end RAG orchestration: ingest PDFs, retrieve, and answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .ingestion import ingest_pdf
from .llm import AnswerService, Citation
from .vectorstore import RetrievedChunk, VectorStore


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunks_added: int


@dataclass(frozen=True)
class AskResult:
    question: str
    answer: str
    grounded: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore | None = None,
        answerer: AnswerService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or VectorStore(self.settings.chroma_dir)
        self.answerer = answerer or AnswerService()

    def ingest(self, pdf_path: Path) -> IngestResult:
        chunks = ingest_pdf(
            pdf_path,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        added = self.store.add(chunks)
        return IngestResult(source=pdf_path.name, chunks_added=added)

    def ask(self, question: str, k: int | None = None) -> AskResult:
        retrieved = self.store.search(question, k=k or self.settings.top_k)
        result = self.answerer.answer(question, retrieved)
        return AskResult(
            question=question,
            answer=result.answer,
            grounded=result.grounded,
            citations=result.citations,
            retrieved=retrieved,
        )
