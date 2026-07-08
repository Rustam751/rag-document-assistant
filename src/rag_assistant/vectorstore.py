"""ChromaDB-backed vector store for document chunks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import chromadb

from .ingestion import Chunk

COLLECTION_NAME = "documents"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page: int
    score: float  # cosine similarity in [0, 1] (higher is more relevant)


class VectorStore:
    """Thin wrapper over a persistent Chroma collection.

    An ``embedding_function`` can be injected (e.g. a deterministic one in tests);
    by default Chroma's built-in ONNX all-MiniLM-L6-v2 model is used, which runs
    locally with no API key.
    """

    def __init__(self, persist_dir: str, embedding_function=None) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        kwargs = {"metadata": {"hnsw:space": "cosine"}}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME, **kwargs)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=str(meta["source"]),
                    page=int(meta["page"]),
                    score=round(1.0 - float(distance), 4),
                )
            )
        return retrieved

    def iter_chunks(self):
        """Yield (chunk_id, text, metadata) for every stored chunk (debug/inspection)."""
        data = self._collection.get(include=["documents", "metadatas"])
        yield from zip(data["ids"], data["documents"], data["metadatas"], strict=True)

    def list_sources(self) -> dict[str, int]:
        """Return {source_file: chunk_count} for every ingested document."""
        data = self._collection.get(include=["metadatas"])
        return dict(Counter(str(m["source"]) for m in data["metadatas"]))

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def count(self) -> int:
        return self._collection.count()
