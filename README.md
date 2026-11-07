# Audio8 ASR Server

A lightweight, **CPU-only** Automatic Speech Recognition service powered by [Audio8-ASR-0.1B](https://huggingface.co/Edge0/Audio8-ASR-0.1B).

Built with FastAPI + ONNX Runtime. Designed for easy Docker deployment (similar in spirit to [SenseVoice-ASR-Server](https://github.com/DaDao-Labs/SenseVoice-ASR-Server)).

**Docker Hub:** [`wereling1992/audio8-asr-server`](https://hub.docker.com/r/wereling1992/audio8-asr-server)

## Features

- **Simple REST API** – `POST /asr` for file upload transcription
- **CPU-only** – no GPU required
- **API Key authentication**
- **Hotword boosting** at decode time
- **Multiple precision options** – int8 (default) / int4 / fp32
- **Docker ready** – `docker pull` + `docker run`
- **OpenAI-compatible API** – `POST /v1/audio/transcriptions`
- **Simple Web UI**
- **Multilingual** – Chinese, English, Cantonese, French, German, Japanese, Korean

## Quick Start (pre-built image)

```bash
docker run -d \
  -p 8000:8000 \
  -e AUDIO8_API_KEY=MySuperSafeApiKey \
  -v audio8-models:/models \
  --name audio8-asr \
  wereling1992/audio8-asr-server:latest
```

First start downloads the ≈3 GB model into the `audio8-models` volume (only once).

### Health check

```bash
curl http://localhost:8000/health
# → {"status":"ok","ok":true, ...}
```

### Transcribe

```bash
curl -X POST http://localhost:8000/asr \
  -H "x-api-key: MySuperSafeApiKey" \
  -F "audio=@your.wav" \
  -F "hotwords=Audio8,Edge0"
```

Web UI: http://localhost:8000/

### OpenAI-compatible API

```bash
# curl
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer MySuperSafeApiKey" \
  -F "file=@your.wav" \
  -F "model=audio8" \
  -F "response_format=json"
# → {"text":"..."}
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="MySuperSafeApiKey")
with open("your.wav", "rb") as f:
    r = client.audio.transcriptions.create(model="audio8", file=f)
print(r.text)
```

Supported `response_format`: `json` (default), `text`, `verbose_json`.  
Auth: `Authorization: Bearer <key>` **or** `x-api-key: <key>`.


---

## Build from source (optional)

```bash
cd Audio8-ASR-Server
docker compose up -d --build
```

Or plain Docker:

```bash
docker build -t wereling1992/audio8-asr-server:latest .
docker run -d -p 8000:8000 -e AUDIO8_API_KEY=MySuperSafeApiKey \
  -v audio8-models:/models --name audio8-asr \
  wereling1992/audio8-asr-server:latest
```

---

## Publish to Docker Hub (maintainer)

```bash
cd Audio8-ASR-Server

# 1. Build
docker build -t wereling1992/audio8-asr-server:latest .

# 2. Login
docker login

# 3. Push
docker push wereling1992/audio8-asr-server:latest

# Optional: also tag a version
docker tag wereling1992/audio8-asr-server:latest wereling1992/audio8-asr-server:1.0.0
docker push wereling1992/audio8-asr-server:1.0.0
```

After push, anyone can use the **Quick Start** `docker run` above without building.

---

## Configuration

| Environment variable | Description | Default |
|----------------------|-------------|---------|
| `AUDIO8_API_KEY` | API key for `x-api-key` header | `changeme` |
| `AUDIO8_REQUIRE_API_KEY` | Enforce API key (`0`/`false` to disable) | `1` |
| `ASR_CACHE_PRECISION` | `int8` / `int4` / `fp32` | `int8` |
| `ASR_AUDIO_PRECISION` | `int8` / `fp32` | `int8` |
| `ASR_BACKEND` | `auto` / `onnx_cache` / `onnx` | `auto` |
| `ORT_PROVIDER` | ONNX Runtime provider | `CPUExecutionProvider` |
| `PORT` | Host port mapping | `8000` |
| `FORCE_DOWNLOAD` | Force re-download of model | `0` |

## Local Development (without Docker)

```bash
git lfs install
git clone https://huggingface.co/Edge0/Audio8-ASR-0.1B-onnx-runtime
export AUDIO8_ASR_BUNDLE=./Audio8-ASR-0.1B-onnx-runtime/model_bundle
export AUDIO8_API_KEY=devkey

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

## API

### `GET /health`
No auth. Returns service status.

### `POST /asr`
Header: `x-api-key: <key>`

| Form field | Type | Description |
|------------|------|-------------|
| `audio` | file | Audio file (wav/mp3/flac/ogg/m4a…) |
| `hotwords` | string | Optional comma-separated boost terms |
| `max_new_tokens` | int | Generation limit (default 128) |
| `cache_precision` | string | Override: `int8` / `int4` / `fp32` |
| `audio_precision` | string | Override: `int8` / `fp32` |

## Resource Requirements

- **RAM**: ~1.5–2.5 GB peak (int8)
- **Disk**: ~3.5 GB for model + runtime
- **CPU**: any x86_64 / ARM64
- **GPU**: not required

## License & Attribution

- Model weights & original ONNX runtime: **CC BY-NC 4.0**  
  © Edge0 / AutoArk-AI – https://huggingface.co/Edge0/Audio8-ASR-0.1B
- Non-commercial use only for the model weights.

## Acknowledgements

- [Edge0 / AutoArk-AI](https://edge0.ai) for Audio8-ASR-0.1B
- Official package: [Edge0/Audio8-ASR-0.1B-onnx-runtime](https://huggingface.co/Edge0/Audio8-ASR-0.1B-onnx-runtime)
- Inspired by [DaDao-Labs/SenseVoice-ASR-Server](https://github.com/DaDao-Labs/SenseVoice-ASR-Server)
