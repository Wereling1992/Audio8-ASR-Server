FROM python:3.12-slim-bookworm

# System deps for audio + onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY asr_onnx_runtime.py .
COPY server.py .
COPY hotword/ ./hotword/
COPY static/ ./static/
COPY scripts/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Model will be downloaded / mounted at runtime
ENV AUDIO8_ASR_BUNDLE=/models/model_bundle
ENV HOST=0.0.0.0
ENV PORT=8000
ENV ASR_BACKEND=auto
ENV ASR_CACHE_PRECISION=int8
ENV ASR_AUDIO_PRECISION=int8
ENV ORT_PROVIDER=CPUExecutionProvider
ENV AUDIO8_TIMEOUT=120

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "server.py"]
