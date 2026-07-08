FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .[ui]

COPY app ./app
COPY eval ./eval

# Vector store + uploads live here; mount a volume to persist
ENV CHROMA_DIR=/data/chroma \
    UPLOAD_DIR=/data/uploads
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "rag_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
