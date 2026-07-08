"""Shared test fixtures: deterministic embeddings, fake LLM client, temp vector store."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from rag_assistant.vectorstore import VectorStore

DIM = 64


class DeterministicEmbedding(EmbeddingFunction):
    """Bag-of-words hashing embedding — deterministic, offline, similarity-preserving
    enough for retrieval tests (shared words => closer vectors)."""

    def __init__(self) -> None:  # required by newer chromadb EF interface
        pass

    @staticmethod
    def name() -> str:
        return "deterministic-test-embedding"

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * DIM
            for word in text.lower().split():
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
                vec[bucket] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(str(tmp_path / "chroma"), embedding_function=DeterministicEmbedding())


class FakeAnthropicClient:
    """Mimics anthropic.Anthropic().messages.create for offline tests."""

    def __init__(self, payload: dict | str, stop_reason: str = "end_turn"):
        text = payload if isinstance(payload, str) else json.dumps(payload)
        block = SimpleNamespace(type="text", text=text)
        response = SimpleNamespace(content=[block], stop_reason=stop_reason)
        self.calls: list[dict] = []

        def create(**kwargs):
            self.calls.append(kwargs)
            return response

        self.messages = SimpleNamespace(create=create)
