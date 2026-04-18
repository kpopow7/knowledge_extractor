# RAG HTTP API — Python 3.12 slim (PyMuPDF wheels; optional PyTorch via sentence-transformers is heavy).
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RAG_STORAGE_ROOT=/data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_api/ ./rag_api/
COPY rag_chunker/ ./rag_chunker/
COPY rag_eval/ ./rag_eval/
COPY rag_extractor/ ./rag_extractor/
COPY rag_generate/ ./rag_generate/
COPY rag_index/ ./rag_index/
COPY rag_retrieve/ ./rag_retrieve/
COPY rag_worker/ ./rag_worker/
COPY rag_storage/ ./rag_storage/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "rag_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
