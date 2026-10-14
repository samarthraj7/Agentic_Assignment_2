# Multi-stage Cloud Run image. Python 3.13-slim, non-root, PORT-bound uvicorn.
# The Chroma index is baked in so the service can retrieve without a download on cold start.

FROM python:3.13-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.13-slim
WORKDIR /app
RUN useradd -m appuser
COPY --from=build /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    EMBEDDING_CACHE_DIR=/home/appuser/.cache/fastembed \
    CHROMA_PERSIST_DIR=/app/chroma_index \
    STATE_BACKEND=sqlite \
    SQLITE_STATE_PATH=/tmp/state.sqlite

COPY . .
# Pre-warm the ONNX embedding model so the first retrieval does not download it.
RUN EMBEDDING_CACHE_DIR=/home/appuser/.cache/fastembed \
    python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/home/appuser/.cache/fastembed')" \
    && chown -R appuser:appuser /app /home/appuser
USER appuser
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
