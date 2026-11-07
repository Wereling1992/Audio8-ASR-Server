#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${AUDIO8_ASR_BUNDLE:-/models/model_bundle}"
REPO_ID="${AUDIO8_REPO_ID:-Edge0/Audio8-ASR-0.1B-onnx-runtime}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

mkdir -p "$(dirname "$MODEL_DIR")"

need_download=0
if [[ ! -f "$MODEL_DIR/metadata.json" ]]; then
  need_download=1
fi
if [[ "$FORCE_DOWNLOAD" == "1" || "$FORCE_DOWNLOAD" == "true" ]]; then
  need_download=1
  rm -rf "$MODEL_DIR"
fi

if [[ "$need_download" == "1" ]]; then
  echo "[entrypoint] Downloading model bundle from Hugging Face: $REPO_ID ..."
  echo "[entrypoint] This may take several minutes (≈3 GB). Please be patient."
  python - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ.get("AUDIO8_REPO_ID", "Edge0/Audio8-ASR-0.1B-onnx-runtime")
target = Path(os.environ.get("AUDIO8_ASR_BUNDLE", "/models/model_bundle"))
tmp_dir = target.parent / "_download_tmp"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    allow_patterns=["model_bundle/**"],
    local_dir=str(tmp_dir),
    local_dir_use_symlinks=False,
)

src = tmp_dir / "model_bundle"
if not src.exists():
    candidates = list(tmp_dir.rglob("metadata.json"))
    if candidates:
        src = candidates[0].parent
    else:
        raise SystemExit(f"Could not find model_bundle after download. Contents: {list(tmp_dir.iterdir())}")

if target.exists():
    shutil.rmtree(target)
shutil.move(str(src), str(target))
shutil.rmtree(tmp_dir, ignore_errors=True)
print(f"[entrypoint] Model ready at {target}")
PY
else
  echo "[entrypoint] Model bundle already present at $MODEL_DIR"
fi

if [[ ! -f "$MODEL_DIR/metadata.json" ]]; then
  echo "[entrypoint] ERROR: metadata.json not found under $MODEL_DIR" >&2
  ls -la "$(dirname "$MODEL_DIR")" || true
  exit 1
fi

echo "[entrypoint] Starting Audio8 ASR Server on ${HOST:-0.0.0.0}:${PORT:-8000}"
exec "$@"
