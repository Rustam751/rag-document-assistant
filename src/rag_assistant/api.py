"""FastAPI application exposing the RAG pipeline."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .llm import MissingAPIKeyError
from .pipeline import RAGPipeline

app = FastAPI(
    title="RAG Document Assistant",
    description="Grounded question answering over PDF documents with cited sources.",
    version=__version__,
)


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    source_index: int
    quote: str
    source: str
    page: int


class RetrievedOut(BaseModel):
    source: str
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    citations: list[CitationOut]
    retrieved: list[RetrievedOut]


class IngestResponse(BaseModel):
    source: str
    chunks_added: int


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/documents")
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    return {"documents": pipeline.store.list_sources(), "total_chunks": pipeline.store.count()}


@app.post("/documents", response_model=IngestResponse)
def upload_document(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        result = pipeline.ingest(dest)
    except Exception as exc:  # corrupt / unreadable PDF
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to ingest PDF: {exc}") from exc

    if result.chunks_added == 0:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found (the PDF may be scanned images — OCR it first).",
        )
    return IngestResponse(source=result.source, chunks_added=result.chunks_added)


@app.delete("/documents/{source}")
def delete_document(source: str, pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    sources = pipeline.store.list_sources()
    if source not in sources:
        raise HTTPException(status_code=404, detail=f"Document '{source}' not found.")
    pipeline.store.delete_source(source)
    return {"deleted": source}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
    try:
        result = pipeline.ask(request.question, k=request.top_k)
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except anthropic.AuthenticationError as exc:
        raise HTTPException(
            status_code=503, detail="ANTHROPIC_API_KEY is missing or invalid."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(status_code=429, detail="Model rate limit hit; retry shortly.") from exc
    except anthropic.APIStatusError as exc:
        # Surface the provider's own message (e.g. insufficient credits) instead of a bare 500.
        raise HTTPException(status_code=502, detail=f"Model API error: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(status_code=503, detail="Could not reach the model API.") from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        citations=[CitationOut(**vars(c)) for c in result.citations],
        retrieved=[
            RetrievedOut(source=r.source, page=r.page, score=r.score, text=r.text)
            for r in result.retrieved
        ],
    )
