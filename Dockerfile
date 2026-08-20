# CPU serve image. The index and ONNX graphs are copied from the build
# context (this laptop), not rebuilt. Needs ~4 GB RAM at runtime.
# Fly VMs in this app are amd64 (fly.toml). An arm64 image fails with
# exec format error on boot.
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    RAGOA_ENCODER=onnx \
    RAGOA_RERANK=0 \
    RAGOA_STT=1 \
    RAGOA_INDEX_DIR=data/index \
    CFLAGS="-O2 -march=x86-64 -mtune=generic" \
    CXXFLAGS="-O2 -march=x86-64 -mtune=generic" \
    NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512_SKX,AVX512CD,AVX512BW,AVX512DQ,AVX512VL"

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY ragoa ragoa
COPY apps/api apps/api
COPY apps/__init__.py apps/__init__.py
# Longer pip timeout: pyarrow is ~47 MB and has timed out mid-download.
RUN pip install --default-timeout=180 --retries 20 --no-cache-dir -e ".[serve]"

COPY data/index data/index
COPY data/onnx data/onnx

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
