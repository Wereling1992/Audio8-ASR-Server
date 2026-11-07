from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
import threading
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
import psutil

from asr_onnx_runtime import OnnxAsrEngine, OnnxCacheAsrEngine, collect_metrics


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
ENGINE: OnnxAsrEngine | None = None
ENGINE_INFO: dict[str, Any] = {}
ENGINE_ARGS: argparse.Namespace | None = None
ENGINE_LOCK = threading.RLock()
PROCESS = psutil.Process()
SERVICE_PEAK_RSS = PROCESS.memory_info().rss
LAST_REQUEST_PEAK_RSS = SERVICE_PEAK_RSS

API_KEY = os.environ.get("AUDIO8_API_KEY", "changeme").strip()
REQUIRE_API_KEY = os.environ.get("AUDIO8_REQUIRE_API_KEY", "1").lower() not in {"0", "false", "no", "off"}

app = FastAPI(
    title="Audio8 ASR Server",
    description="Lightweight CPU-based multilingual ASR service powered by Audio8-ASR-0.1B. "
    "OpenAI-compatible endpoint: POST /v1/audio/transcriptions",
    version="1.1.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def release_engine() -> None:
    global ENGINE
    ENGINE = None
    gc.collect()


class RequestMemorySampler:
    def __init__(self, sample_interval_s: float = 0.05) -> None:
        self.sample_interval_s = float(sample_interval_s)
        self.peak_rss_bytes = PROCESS.memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            self.peak_rss_bytes = max(self.peak_rss_bytes, PROCESS.memory_info().rss)

    def start(self) -> None:
        self.peak_rss_bytes = PROCESS.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self.peak_rss_bytes = max(self.peak_rss_bytes, PROCESS.memory_info().rss)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_rss_bytes


def update_service_peak(candidate: int | None = None) -> None:
    global SERVICE_PEAK_RSS
    current = PROCESS.memory_info().rss
    values = [SERVICE_PEAK_RSS, current]
    if candidate is not None:
        values.append(int(candidate))
    SERVICE_PEAK_RSS = max(values)


def available_cache_precisions(bundle_dir: str | Path) -> list[str]:
    bundle_path = Path(bundle_dir).expanduser().resolve()
    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prefill_graph = metadata.get("graphs", {}).get("lm_cache_prefill")
    graph = metadata.get("graphs", {}).get("lm_cache_decode")
    if not graph or not prefill_graph:
        return []
    prefill_graph_path = bundle_path / prefill_graph["path"]
    graph_path = bundle_path / graph["path"]
    values = []
    if prefill_graph_path.exists() and graph_path.exists():
        values.append("fp32")
    prefill_int8_path = prefill_graph_path.with_name(f"{prefill_graph_path.stem}_int8{prefill_graph_path.suffix}")
    int8_path = graph_path.with_name(f"{graph_path.stem}_int8{graph_path.suffix}")
    if prefill_int8_path.exists() and int8_path.exists():
        values.append("int8")
    prefill_int4_path = prefill_graph_path.with_name(f"{prefill_graph_path.stem}_int4{prefill_graph_path.suffix}")
    int4_path = graph_path.with_name(f"{graph_path.stem}_int4{graph_path.suffix}")
    if prefill_int4_path.exists() and int4_path.exists():
        values.append("int4")
    return values


def available_audio_precisions(bundle_dir: str | Path) -> list[str]:
    bundle_path = Path(bundle_dir).expanduser().resolve()
    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    graphs = metadata.get("graphs", {})
    values = []
    fp32_graph = graphs.get("audio_hidden")
    if fp32_graph and (bundle_path / fp32_graph["path"]).exists():
        values.append("fp32")
    int8_graph = graphs.get("audio_hidden_int8")
    if int8_graph and (bundle_path / int8_graph["path"]).exists():
        values.append("int8")
    return values


def runtime_snapshot() -> dict[str, Any]:
    update_service_peak()
    bundle_dir = ENGINE_INFO.get("bundle_dir")
    return {
        **ENGINE_INFO,
        "runtime_loaded": ENGINE is not None,
        "available_cache_precisions": available_cache_precisions(bundle_dir) if bundle_dir else [],
        "available_audio_precisions": available_audio_precisions(bundle_dir) if bundle_dir else [],
        "service_peak_rss_bytes": int(SERVICE_PEAK_RSS),
        "last_request_peak_rss_bytes": int(LAST_REQUEST_PEAK_RSS),
    }


def load_engine(
    args: argparse.Namespace,
    *,
    backend: str | None = None,
    cache_precision: str | None = None,
    audio_precision: str | None = None,
    release_existing: bool = False,
) -> None:
    global ENGINE, ENGINE_INFO
    if release_existing:
        release_engine()
    selected_backend = backend or args.backend
    selected_precision = cache_precision or args.cache_precision
    selected_audio_precision = audio_precision if audio_precision is not None else getattr(args, "audio_precision", None)
    threads = args.threads if args.threads > 0 else None

    if selected_backend in {"auto", "onnx_cache"}:
        try:
            engine = OnnxCacheAsrEngine(
                args.bundle_dir,
                provider=args.provider,
                intra_op_num_threads=threads,
                cache_precision=selected_precision,
                audio_precision=selected_audio_precision,
            )
            ENGINE = engine
            ENGINE_INFO = {
                "bundle_dir": str(engine.bundle_dir),
                "backend": "onnx_cache",
                "cache_precision": engine.cache_precision,
                "cache_prefill_graph": str(engine.prefill_graph_path),
                "cache_graph": str(engine.cache_graph_path),
                "audio_precision": engine.audio_precision,
                "audio_graph": str(engine.audio_graph_path),
                "providers": engine.providers,
            }
            update_service_peak()
            return
        except Exception:
            if selected_backend == "onnx_cache":
                raise

    engine = OnnxAsrEngine(
        args.bundle_dir,
        provider=args.provider,
        intra_op_num_threads=threads,
        audio_precision=selected_audio_precision or "fp32",
    )
    ENGINE = engine
    ENGINE_INFO = {
        "bundle_dir": str(engine.bundle_dir),
        "backend": "onnx",
        "cache_precision": None,
        "cache_graph": None,
        "audio_precision": engine.audio_precision,
        "audio_graph": str(engine.audio_graph_path),
        "providers": engine.providers,
    }
    update_service_peak()


# ---------- Auth (x-api-key or Authorization: Bearer) ----------

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None),
) -> None:
    if not REQUIRE_API_KEY:
        return
    token = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        else:
            token = authorization.strip()
    if not token or token != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Use header x-api-key or Authorization: Bearer <key>",
        )


# ---------- Shared transcription core ----------

def _maybe_reload_precision(
    cache_precision: str | None,
    audio_precision: str | None,
) -> None:
    if ENGINE_ARGS is None:
        return
    requested_cache = cache_precision.lower().strip() if cache_precision else None
    requested_audio = audio_precision.lower().strip() if audio_precision else None
    if requested_cache and requested_cache not in {"fp32", "int8", "int4", "auto"}:
        raise HTTPException(status_code=400, detail=f"Unsupported cache precision: {requested_cache}")
    if requested_audio and requested_audio not in {"fp32", "int8", "auto"}:
        raise HTTPException(status_code=400, detail=f"Unsupported audio precision: {requested_audio}")
    current_cache = str(ENGINE_INFO.get("cache_precision") or "").lower().strip()
    current_audio = str(ENGINE_INFO.get("audio_precision") or "").lower().strip()
    target_cache = requested_cache or current_cache or ENGINE_ARGS.cache_precision
    target_audio = requested_audio or current_audio or getattr(ENGINE_ARGS, "audio_precision", None)
    should_reload = ENGINE_INFO.get("backend") == "onnx_cache" and (
        (requested_cache is not None and target_cache != current_cache)
        or (requested_audio is not None and target_audio != current_audio)
    )
    if should_reload:
        with ENGINE_LOCK:
            try:
                load_engine(
                    ENGINE_ARGS,
                    backend="onnx_cache",
                    cache_precision=target_cache,
                    audio_precision=target_audio,
                    release_existing=True,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_transcription(
    audio_bytes: bytes,
    *,
    language: str | None = None,
    max_new_tokens: int = 128,
    cache_precision: str | None = None,
    audio_precision: str | None = None,
    hotwords: str | None = None,
    hotword_topk: int = 50,
    hotword_start_boost: float = 6.0,
    hotword_continuation_boost: float = 8.0,
) -> dict[str, Any]:
    global LAST_REQUEST_PEAK_RSS
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="ASR engine is not loaded")
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    _maybe_reload_precision(cache_precision, audio_precision)

    sampler = RequestMemorySampler()
    sampler.start()
    start_t = time.perf_counter()
    try:
        with ENGINE_LOCK:
            if ENGINE is None:
                raise HTTPException(status_code=503, detail="ASR engine is not loaded")
            result = ENGINE.transcribe(
                audio_bytes,
                language=language or None,
                max_new_tokens=max_new_tokens,
                hotwords=hotwords or None,
                hotword_topk=int(hotword_topk),
                hotword_start_boost=float(hotword_start_boost),
                hotword_continuation_boost=float(hotword_continuation_boost),
            )
        elapsed = time.perf_counter() - start_t
        LAST_REQUEST_PEAK_RSS = max(int(sampler.stop()), PROCESS.memory_info().rss)
        update_service_peak(LAST_REQUEST_PEAK_RSS)

        text = result.get("text") or result.get("prediction") or ""
        return {
            "text": text,
            "raw": result.get("raw"),
            "elapsed_seconds": round(elapsed, 3),
            "audio_seconds": result.get("audio_seconds"),
            "generated_tokens": result.get("generated_tokens"),
            "backend": result.get("backend") or ENGINE_INFO.get("backend"),
            "cache_precision": result.get("cache_precision") or ENGINE_INFO.get("cache_precision"),
            "audio_precision": result.get("audio_precision") or ENGINE_INFO.get("audio_precision"),
            "request_peak_rss_bytes": int(LAST_REQUEST_PEAK_RSS),
        }
    except HTTPException:
        raise
    except Exception as exc:
        LAST_REQUEST_PEAK_RSS = max(int(sampler.stop()), PROCESS.memory_info().rss)
        update_service_peak(LAST_REQUEST_PEAK_RSS)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------- Routes ----------

@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "service": "Audio8 ASR Server",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "asr": "POST /asr",
        "openai_compatible": "POST /v1/audio/transcriptions",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    if ENGINE is None:
        return {"status": "error", "ok": False, "error": "engine not loaded"}
    return {"status": "ok", "ok": True, **runtime_snapshot()}


@app.get("/api/runtime")
async def runtime(_: None = Depends(verify_api_key)) -> dict[str, Any]:
    return {"ok": ENGINE is not None, "runtime": runtime_snapshot()}


@app.post("/api/reload")
async def reload_runtime(
    backend: str = Form("onnx_cache"),
    cache_precision: str | None = Form(None),
    audio_precision: str | None = Form(None),
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    if ENGINE_ARGS is None:
        raise HTTPException(status_code=503, detail="Server arguments are not initialized")
    if backend not in {"onnx_cache", "onnx", "auto"}:
        raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")
    if cache_precision:
        cache_precision = cache_precision.lower().strip()
        if cache_precision not in {"fp32", "int8", "int4", "auto"}:
            raise HTTPException(status_code=400, detail=f"Unsupported cache precision: {cache_precision}")
    else:
        cache_precision = str(ENGINE_INFO.get("cache_precision") or "").lower().strip() or ENGINE_ARGS.cache_precision
    if audio_precision:
        audio_precision = audio_precision.lower().strip()
        if audio_precision not in {"fp32", "int8", "auto"}:
            raise HTTPException(status_code=400, detail=f"Unsupported audio precision: {audio_precision}")
    else:
        audio_precision = str(ENGINE_INFO.get("audio_precision") or "").lower().strip() or None
    with ENGINE_LOCK:
        try:
            load_engine(
                ENGINE_ARGS,
                backend=backend,
                cache_precision=cache_precision,
                audio_precision=audio_precision,
                release_existing=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "runtime": runtime_snapshot()}


@app.get("/metrics")
async def metrics(_: None = Depends(verify_api_key)) -> dict[str, Any]:
    data = collect_metrics()
    data["runtime"] = runtime_snapshot()
    return data


@app.post("/asr")
async def asr(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    max_new_tokens: int = Form(128),
    cache_precision: str | None = Form(None),
    audio_precision: str | None = Form(None),
    hotwords: str | None = Form(None),
    hotword_topk: int = Form(50),
    hotword_start_boost: float = Form(6.0),
    hotword_continuation_boost: float = Form(8.0),
    _: None = Depends(verify_api_key),
) -> dict[str, Any]:
    audio_bytes = await audio.read()
    result = run_transcription(
        audio_bytes,
        language=language,
        max_new_tokens=max_new_tokens,
        cache_precision=cache_precision,
        audio_precision=audio_precision,
        hotwords=hotwords,
        hotword_topk=hotword_topk,
        hotword_start_boost=hotword_start_boost,
        hotword_continuation_boost=hotword_continuation_boost,
    )
    return {"code": 0, **result}


# ---------- OpenAI-compatible API ----------

@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
    # Extra (non-OpenAI) fields still accepted
    max_new_tokens: int = Form(128),
    hotwords: str | None = Form(None),
    _: None = Depends(verify_api_key),
):
    """
    OpenAI-compatible transcription endpoint.

    Compatible with OpenAI Python SDK::

        from openai import OpenAI
        client = OpenAI(base_url="http://host:8000/v1", api_key="your-key")
        with open("audio.wav", "rb") as f:
            r = client.audio.transcriptions.create(model="audio8", file=f)
        print(r.text)
    """
    del model, temperature  # accepted for compatibility, not used by Audio8

    # OpenAI "prompt" is often used as context / hotword-like hint
    effective_hotwords = hotwords
    if prompt and not effective_hotwords:
        effective_hotwords = prompt

    audio_bytes = await file.read()
    result = run_transcription(
        audio_bytes,
        language=language,
        max_new_tokens=max_new_tokens,
        hotwords=effective_hotwords,
    )
    text = result.get("text") or ""
    fmt = (response_format or "json").lower().strip()

    if fmt == "text":
        return PlainTextResponse(text)

    if fmt == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or "unknown",
            "duration": result.get("audio_seconds"),
            "text": text,
            "segments": [],
            # Extra diagnostics (non-breaking for most clients)
            "elapsed_seconds": result.get("elapsed_seconds"),
            "backend": result.get("backend"),
            "cache_precision": result.get("cache_precision"),
        }

    # default: json  →  {"text": "..."}
    return {"text": text}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio8 ASR Server")
    parser.add_argument("--bundle_dir", default=os.environ.get("AUDIO8_ASR_BUNDLE", "model_bundle"))
    parser.add_argument("--backend", choices=["auto", "onnx", "onnx_cache"], default=os.environ.get("ASR_BACKEND", "auto"))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--provider", default=os.environ.get("ORT_PROVIDER", "CPUExecutionProvider"))
    parser.add_argument("--cache_precision", default=os.environ.get("ASR_CACHE_PRECISION", "int8"))
    parser.add_argument("--audio_precision", default=os.environ.get("ASR_AUDIO_PRECISION", "int8"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("ORT_THREADS", "0")))
    return parser.parse_args()


def main() -> None:
    global ENGINE_ARGS
    args = parse_args()
    ENGINE_ARGS = args

    print(f"[startup] Loading engine from {args.bundle_dir} ...")
    load_engine(args)
    print(
        f"[startup] Engine ready – backend={ENGINE_INFO.get('backend')} "
        f"cache={ENGINE_INFO.get('cache_precision')} audio={ENGINE_INFO.get('audio_precision')}"
    )
    print(f"[startup] API key required: {REQUIRE_API_KEY}")
    print("[startup] OpenAI-compatible: POST /v1/audio/transcriptions")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
