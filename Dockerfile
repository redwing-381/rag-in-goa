# CPU serve image. The index and ONNX graphs are copied from the build
# context (this laptop), not rebuilt. Needs ~4 GB RAM at runtime.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    RAGOA_ENCODER=onnx \
    RAGOA_RERANK=0 \
    RAGOA_STT=1 \
    RAGOA_INDEX_DIR=data/index

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY ragoa ragoa
COPY apps/api apps/api
COPY apps/__init__.py apps/__init__.py
RUN pip install --no-cache-dir -e ".[serve]"

COPY data/index data/index
COPY data/onnx data/onnx

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
