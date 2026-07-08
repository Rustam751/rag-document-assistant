"""Grounded answer generation with Claude.

The model receives numbered source excerpts and must answer strictly from them,
returning structured JSON with per-claim citations (hallucination guard).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import anthropic

from .config import get_settings
from .vectorstore import RetrievedChunk

SYSTEM_PROMPT = """\
You are a document question-answering assistant. You answer questions strictly \
from the numbered source excerpts provided in each request.

Rules:
- Use ONLY the provided sources. Never use outside knowledge, even if you know the answer.
- Every factual claim in your answer must be supported by at least one cited source.
- Cite sources by their number, and include a short verbatim quote from the source \
that supports the claim.
- If the sources do not contain enough information to answer, say so plainly in the \
answer field and return an empty citations list. Do not guess.
- Keep answers concise and factual.\
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the question, based only on the provided sources.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "integer",
                        "description": "1-based index of the supporting source excerpt.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Short verbatim supporting quote from that source.",
                    },
                },
                "required": ["source", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Citation:
    source_index: int  # 1-based index into the retrieved chunks
    quote: str
    source: str = ""  # resolved file name
    page: int = 0  # resolved page number


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True  # False when the model could not answer from the sources


def format_sources(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({chunk.source}, page {chunk.page})\n{chunk.text}")
    return "\n\n".join(lines)


class MissingAPIKeyError(RuntimeError):
    """Raised when no Anthropic API key is configured (neither .env nor environment)."""


class AnswerService:
    """Calls Claude to produce a grounded, cited answer from retrieved chunks."""

    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        settings = get_settings()
        self._client = client  # constructed lazily so the app can boot without a key
        self._api_key = settings.anthropic_api_key
        self._model = model or settings.anthropic_model
        self._max_tokens = settings.max_answer_tokens

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key or api_key.startswith("sk-ant-..."):
                raise MissingAPIKeyError(
                    "No Anthropic API key configured. Set ANTHROPIC_API_KEY in .env "
                    "(get a key at https://console.anthropic.com/settings/keys)."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
        if not chunks:
            return GroundedAnswer(
                answer="No documents have been ingested yet, or nothing relevant was found.",
                grounded=False,
            )

        prompt = (
            f"Sources:\n\n{format_sources(chunks)}\n\n"
            f"Question: {question}\n\n"
            "Answer the question using only the sources above."
        )

        response = self.client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        )

        if response.stop_reason == "refusal":
            return GroundedAnswer(answer="The request was declined by the model.", grounded=False)
        if response.stop_reason == "max_tokens":
            return GroundedAnswer(
                answer="The answer was truncated; try a more specific question.",
                grounded=False,
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Schema-constrained output should always parse; degrade gracefully if not.
            return GroundedAnswer(answer=text, grounded=False)

        citations = []
        for item in data.get("citations", []):
            idx = int(item["source"])
            resolved = chunks[idx - 1] if 1 <= idx <= len(chunks) else None
            citations.append(
                Citation(
                    source_index=idx,
                    quote=item["quote"],
                    source=resolved.source if resolved else "",
                    page=resolved.page if resolved else 0,
                )
            )
        return GroundedAnswer(
            answer=data["answer"],
            citations=citations,
            grounded=bool(citations),
        )
