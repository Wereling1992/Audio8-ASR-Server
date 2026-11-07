from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
from tokenizers import Tokenizer
from transformers import WhisperFeatureExtractor

from hotword.hotword_trie import build_trie_from_hotwords, parse_hotwords


SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|(?:"
    r"bicodec_(?:semantic|global)_\d+|"
    r"(?:start|end)_(?:global_token|glm_token|semantic_token|content)|"
    r"[^>]+"
    r")\|>"
)
TURN_END_MARKERS = ("<|user|>", "<|assistant|>", "<|im_end|>")
LEADING_NOISE_PATTERN = re.compile(r"^[\s,.;:!?-]+")


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = float(audio.shape[0]) / float(orig_sr)
    target_len = max(1, int(round(duration * float(target_sr))))
    old_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32, copy=False)


def load_audio_bytes(audio_bytes: bytes, sampling_rate: int) -> np.ndarray:
    try:
        import librosa

        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=int(sampling_rate), mono=True)
        return np.asarray(audio, dtype=np.float32)
    except Exception:
        pass

    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        return _resample_linear(np.asarray(audio, dtype=np.float32), int(sr), int(sampling_rate))
    except Exception:
        pass

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        sr = int(wav.getframerate())
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        raw = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"Fallback wave loader only supports 16-bit PCM WAV, got sample_width={sample_width}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=-1)
    return _resample_linear(audio, sr, int(sampling_rate))


def truncate_generation_text(text: str) -> str:
    cut = len(text)
    for marker in TURN_END_MARKERS:
        index = text.find(marker)
        if index != -1 and index < cut:
            cut = index
    return text[:cut].strip()


def normalize_prediction_text(text: str) -> str:
    if not text:
        return ""
    text = truncate_generation_text(text)
    if "<|text|>" in text:
        text = text.split("<|text|>", 1)[1]
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    text = re.sub(r"^\s*language\s+[A-Za-z]+\s+", "", text)
    text = SPECIAL_TOKEN_PATTERN.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return LEADING_NOISE_PATTERN.sub("", text).strip()


def create_ort_session_options(intra_op_num_threads: int | None = None) -> ort.SessionOptions:
    options = ort.SessionOptions()
    if intra_op_num_threads is not None and int(intra_op_num_threads) > 0:
        options.intra_op_num_threads = int(intra_op_num_threads)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if os.environ.get("ORT_CPU_MEM_ARENA", "0").lower() not in {"1", "true", "yes", "on"}:
        options.enable_cpu_mem_arena = False
    if os.environ.get("ORT_MEM_PATTERN", "0").lower() not in {"1", "true", "yes", "on"}:
        options.enable_mem_pattern = False
    return options


def ark_audio_token_count(sample_count: int, *, hop_length: int, merge_factor: int) -> int:
    mel_frames = int(sample_count) // max(int(hop_length), 1)
    downsampled = (int(mel_frames) + 1) // 2
    merged = downsampled // max(int(merge_factor), 1)
    return max(int(merged), 1)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    mean = x32.mean(axis=-1, keepdims=True)
    var = ((x32 - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x32 - mean) / np.sqrt(var + eps)) * weight + bias


def adaptive_avg_pool_time(x: np.ndarray, output_size: int) -> np.ndarray:
    input_size = int(x.shape[0])
    output_size = int(output_size)
    if input_size == output_size:
        return x.astype(np.float32, copy=False)
    pooled = np.empty((output_size, x.shape[1]), dtype=np.float32)
    for out_i in range(output_size):
        start = int(math.floor(out_i * input_size / output_size))
        end = int(math.ceil((out_i + 1) * input_size / output_size))
        end = max(end, start + 1)
        pooled[out_i] = x[start:end].mean(axis=0)
    return pooled


def apply_repetition_penalty(logits: np.ndarray, token_ids: list[int], penalty: float) -> np.ndarray:
    if penalty == 1.0:
        return logits
    for token_id in set(int(value) for value in token_ids):
        if 0 <= token_id < logits.shape[-1]:
            logits[token_id] = logits[token_id] * penalty if logits[token_id] < 0 else logits[token_id] / penalty
    return logits


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class OnnxAsrEngine:
    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        provider: str = "CPUExecutionProvider",
        intra_op_num_threads: int | None = None,
        load_lm_session: bool = True,
        audio_precision: str = "fp32",
    ) -> None:
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        with (self.bundle_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.audio_precision = str(audio_precision or "fp32").lower().strip()

        options = create_ort_session_options(intra_op_num_threads)
        providers = [provider] if provider in ort.get_available_providers() else ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        audio_graph_key = "audio_hidden"
        if self.audio_precision in {"int8", "auto"} and "audio_hidden_int8" in self.metadata.get("graphs", {}):
            audio_graph_key = "audio_hidden_int8"
            self.audio_precision = "int8"
        else:
            self.audio_precision = "fp32"
        self.audio_graph = self.metadata["graphs"][audio_graph_key]
        self.audio_graph_path = self.bundle_dir / self.audio_graph["path"]

        self.audio_session = ort.InferenceSession(
            str(self.audio_graph_path),
            sess_options=options,
            providers=providers,
        )
        self.lm_session = None
        if load_lm_session:
            self.lm_session = ort.InferenceSession(
                str(self.bundle_dir / self.metadata["graphs"]["lm_logits"]["path"]),
                sess_options=options,
                providers=providers,
            )
        self.providers = {
            "audio": self.audio_session.get_providers(),
            "lm": self.lm_session.get_providers() if self.lm_session is not None else None,
        }

        self.tokenizer = Tokenizer.from_file(str(self.bundle_dir / "tokenizer.json"))
        feature_dir = self.bundle_dir / "qwen3_asr_feature_extractor"
        if not feature_dir.exists():
            feature_dir = self.bundle_dir
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(feature_dir))
        self.token_embedding = np.load(
            self.bundle_dir / self.metadata["weights"]["token_embedding"],
            mmap_mode="r",
        )
        if self.token_embedding.dtype != np.float32:
            self.token_embedding = self.token_embedding.astype(np.float32, copy=False)
        projector = np.load(self.bundle_dir / self.metadata["weights"]["audio_projector"])
        self.projector = {key: projector[key].astype(np.float32) for key in projector.files}

        tokens = self.metadata["tokens"]
        self.audio_token_id = int(tokens["audio_token_id"])
        self.pad_token_id = int(tokens["pad_token_id"])
        self.eos_token_ids = set(int(value) for value in tokens["eos_token_ids"])
        self.asr_block_token_id_from = int(tokens.get("asr_block_token_id_from", -1))
        self.extra_block_token_ids = set(int(value) for value in tokens.get("extra_block_token_ids", []))
        self.sampling_rate = int(self.metadata["sampling_rate"])
        self.max_audio_seconds = int(self.metadata["max_audio_seconds"])
        prompt_audio = self.metadata.get("prompt_audio", {})
        self.prompt_merge_factor = int(prompt_audio.get("merge_factor") or 4)

    def _token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Token not found in tokenizer: {token}")
        return int(token_id)

    def _build_prompt(self, audio_token_count: int, language: str | None = None) -> str:
        del language
        tokens = self.metadata["tokens"]
        audio_tokens = tokens["audio_token"] * int(audio_token_count)
        return (
            f"{tokens['user_token']}"
            f"{tokens['bos_audio_token']}{audio_tokens}{tokens['eos_audio_token']}"
            "Please transcribe this audio."
            f"{tokens['assistant_token']}"
            f"{self.metadata.get('response_prefix', '') or ''}"
        )

    def _extract_features(self, audio: np.ndarray) -> tuple[np.ndarray, int, int, int]:
        max_samples = int(self.max_audio_seconds * self.sampling_rate)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]
        sample_count = int(max(1, audio.shape[0]))
        feature = self.feature_extractor(
            [audio],
            sampling_rate=self.sampling_rate,
            return_tensors="np",
            return_attention_mask=False,
            padding="longest",
            max_length=max_samples,
        )["input_features"].astype(np.float32)
        hop_length = int(getattr(self.feature_extractor, "hop_length", 160))
        encoder_feature_len = int(math.ceil(float(sample_count) / float(max(hop_length, 1))))
        encoder_feature_len = min(max(1, encoder_feature_len), int(feature.shape[-1]))

        frames_padded = int(self.audio_graph["frames_padded"])
        if feature.shape[-1] < frames_padded:
            feature = np.pad(feature, ((0, 0), (0, 0), (0, frames_padded - feature.shape[-1])), mode="constant")
        elif feature.shape[-1] > frames_padded:
            feature = feature[:, :, :frames_padded]
        return feature.astype(np.float32), sample_count, encoder_feature_len, hop_length

    def _audio_embeddings(
        self,
        feature: np.ndarray,
        sample_count: int,
        encoder_feature_len: int,
        hop_length: int,
    ) -> np.ndarray:
        audio_token_count = ark_audio_token_count(
            sample_count,
            hop_length=hop_length,
            merge_factor=self.prompt_merge_factor,
        )
        hidden, valid_mask = self.audio_session.run(
            None,
            {
                "audios": feature.astype(np.float32, copy=False),
                "audio_feature_lengths": np.asarray([encoder_feature_len], dtype=np.int64),
            },
        )
        hidden = hidden.astype(np.float32, copy=False)
        valid_mask = valid_mask.astype(bool)
        valid_hidden = hidden[valid_mask]
        if valid_hidden.shape[0] != audio_token_count:
            valid_hidden = adaptive_avg_pool_time(valid_hidden, audio_token_count)
        projected = layer_norm(
            valid_hidden,
            self.projector["norm_weight"],
            self.projector["norm_bias"],
        )
        projected = projected @ self.projector["linear_weight"].T + self.projector["linear_bias"]
        return projected.astype(np.float32, copy=False)

    def _initial_embeddings(self, audio_embeddings: np.ndarray, language: str | None) -> tuple[list[int], np.ndarray]:
        prompt = self._build_prompt(audio_embeddings.shape[0], language=language)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        embeds = self.token_embedding[np.asarray(input_ids, dtype=np.int64)].astype(np.float32)
        audio_positions = [index for index, token_id in enumerate(input_ids) if int(token_id) == self.audio_token_id]
        if len(audio_positions) != audio_embeddings.shape[0]:
            raise RuntimeError(
                f"Prompt has {len(audio_positions)} audio tokens, but audio graph returned {audio_embeddings.shape[0]}"
            )
        embeds[np.asarray(audio_positions, dtype=np.int64)] = audio_embeddings
        return [int(value) for value in input_ids], embeds

    def _mask_logits(self, logits: np.ndarray) -> None:
        if self.asr_block_token_id_from >= 0 and self.asr_block_token_id_from < logits.shape[0]:
            logits[self.asr_block_token_id_from :] = -np.inf
        for token_id in self.extra_block_token_ids:
            if 0 <= token_id < logits.shape[0]:
                logits[token_id] = -np.inf

    def _build_hotword_trie(self, hotwords, start_boost: float, continuation_boost: float):
        special_ids = set(self.eos_token_ids)
        special_ids.add(self.pad_token_id)
        special_ids.update(self.extra_block_token_ids)

        def encode(text: str) -> list[int]:
            return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

        def id_to_token(token_id: int) -> str:
            return str(self.tokenizer.id_to_token(int(token_id)) or "")

        trie, sequences_by_word = build_trie_from_hotwords(
            hotwords,
            encode=encode,
            id_to_token=id_to_token,
            special_ids=special_ids,
            start_boost=float(start_boost),
            continuation_boost=float(continuation_boost),
        )
        meta = {
            "hotwords": list(hotwords),
            "hotword_token_ids": {word: variants for word, variants in sequences_by_word.items()},
            "hotword_start_boost": float(start_boost),
            "hotword_continuation_boost": float(continuation_boost),
        }
        return trie, meta

    @staticmethod
    def _apply_hotword_boost(logits: np.ndarray, generated: list[int], trie, topk: int) -> None:
        if not trie:
            return
        boosts = trie.boosts_for_generated(generated)
        if not boosts:
            return
        allowed: set[int] | None = None
        if topk and int(topk) > 0:
            k = min(int(topk), int(logits.shape[-1]))
            allowed = set(int(i) for i in np.argpartition(logits, -k)[-k:])
        vocab = int(logits.shape[-1])
        for token_id, boost in boosts.items():
            if allowed is not None and token_id not in allowed:
                continue
            if 0 <= token_id < vocab:
                logits[token_id] += boost

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.5,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        hotwords: str | list | None = None,
        hotword_topk: int = 50,
        hotword_start_boost: float = 6.0,
        hotword_continuation_boost: float = 8.0,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        audio = load_audio_bytes(audio_bytes, self.sampling_rate)
        feature, sample_count, encoder_feature_len, hop_length = self._extract_features(audio)
        audio_embeddings = self._audio_embeddings(feature, sample_count, encoder_feature_len, hop_length)
        token_ids, embeds = self._initial_embeddings(audio_embeddings, language=language)
        hotword_list = parse_hotwords(hotwords)
        hot_trie, hot_meta = (
            self._build_hotword_trie(hotword_list, hotword_start_boost, hotword_continuation_boost)
            if hotword_list
            else (None, None)
        )

        generated: list[int] = []
        hit_stop = False
        stop_token_id: int | None = None
        rng = np.random.default_rng()
        for _ in range(int(max_new_tokens)):
            if self.lm_session is None:
                raise RuntimeError("ONNX LM session is not loaded")
            attention_mask = np.ones((1, embeds.shape[0]), dtype=np.int64)
            logits = self.lm_session.run(
                None,
                {
                    "inputs_embeds": embeds[None, :, :].astype(np.float32, copy=False),
                    "attention_mask": attention_mask,
                },
            )[0][0].astype(np.float32)
            apply_repetition_penalty(logits, token_ids + generated, float(repetition_penalty))
            self._mask_logits(logits)
            if hot_trie:
                self._apply_hotword_boost(logits, generated, hot_trie, hotword_topk)
            if do_sample:
                probs = softmax(logits / max(float(temperature), 1e-6))
                next_token = int(rng.choice(np.arange(probs.shape[0]), p=probs))
            else:
                next_token = int(np.argmax(logits))
            if next_token in self.eos_token_ids or next_token == self.pad_token_id:
                hit_stop = True
                stop_token_id = next_token
                break
            generated.append(next_token)
            token_embed = self.token_embedding[np.asarray([next_token], dtype=np.int64)].astype(np.float32)
            embeds = np.concatenate([embeds, token_embed], axis=0)

        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        text = normalize_prediction_text(raw)
        elapsed = time.perf_counter() - started
        return {
            "text": text,
            "raw": raw,
            "generated_tokens": len(generated),
            "hit_stop": hit_stop,
            "stop_token_id": stop_token_id,
            "elapsed_seconds": elapsed,
            "audio_seconds": float(audio.shape[0]) / float(self.sampling_rate),
            "audio_token_count": int(audio_embeddings.shape[0]),
            "providers": self.providers,
            "backend": "onnx",
            "audio_precision": self.audio_precision,
            "hotword": hot_meta,
        }


class OnnxCacheAsrEngine(OnnxAsrEngine):
    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        provider: str = "CPUExecutionProvider",
        intra_op_num_threads: int | None = None,
        cache_precision: str = "int8",
        audio_precision: str | None = None,
    ) -> None:
        selected_audio_precision = audio_precision or "fp32"
        super().__init__(
            bundle_dir,
            provider=provider,
            intra_op_num_threads=intra_op_num_threads,
            load_lm_session=False,
            audio_precision=selected_audio_precision,
        )
        prefill_graph = self.metadata.get("graphs", {}).get("lm_cache_prefill")
        graph = self.metadata.get("graphs", {}).get("lm_cache_decode")
        if not graph:
            raise FileNotFoundError("Bundle metadata has no graphs.lm_cache_decode entry")
        if not prefill_graph:
            raise FileNotFoundError("Bundle metadata has no graphs.lm_cache_prefill entry")
        cache_precision = str(cache_precision or "fp32").lower().strip()
        if cache_precision not in {"fp32", "int8", "int4", "auto"}:
            raise ValueError(f"Unsupported cache_precision={cache_precision!r}; use fp32, int8, int4, or auto")
        graph_path = self.bundle_dir / graph["path"]
        prefill_graph_path = self.bundle_dir / prefill_graph["path"]
        int8_path = graph_path.with_name(f"{graph_path.stem}_int8{graph_path.suffix}")
        prefill_int8_path = prefill_graph_path.with_name(f"{prefill_graph_path.stem}_int8{prefill_graph_path.suffix}")
        int4_path = graph_path.with_name(f"{graph_path.stem}_int4{graph_path.suffix}")
        prefill_int4_path = prefill_graph_path.with_name(f"{prefill_graph_path.stem}_int4{prefill_graph_path.suffix}")
        if cache_precision in {"int8", "int4"}:
            requested_paths = (prefill_int8_path, int8_path) if cache_precision == "int8" else (prefill_int4_path, int4_path)
            missing = [str(path) for path in requested_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Requested {cache_precision} cache graph(s) do not exist: {missing}")
            prefill_graph_path, graph_path = requested_paths
        elif cache_precision == "auto":
            if int8_path.exists() and prefill_int8_path.exists():
                graph_path = int8_path
                prefill_graph_path = prefill_int8_path
            elif int4_path.exists() and prefill_int4_path.exists():
                graph_path = int4_path
                prefill_graph_path = prefill_int4_path

        options = create_ort_session_options(intra_op_num_threads)
        providers = [provider] if provider in ort.get_available_providers() else ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        self.prefill_lm_session = ort.InferenceSession(
            str(prefill_graph_path),
            sess_options=options,
            providers=providers,
        )
        self.cache_lm_session = ort.InferenceSession(
            str(graph_path),
            sess_options=options,
            providers=providers,
        )
        self.prefill_graph = prefill_graph
        self.prefill_graph_path = prefill_graph_path
        self.cache_graph = graph
        self.cache_graph_path = graph_path
        if graph_path == int8_path and prefill_graph_path == prefill_int8_path:
            self.cache_precision = "int8"
        elif graph_path == int4_path and prefill_graph_path == prefill_int4_path:
            self.cache_precision = "int4"
        else:
            self.cache_precision = "fp32"
        self.providers["lm"] = self.cache_lm_session.get_providers()
        first_input = self.prefill_lm_session.get_inputs()[0]
        self.lm_embed_dtype = self._ort_type_to_numpy(first_input.type)
        cache_key_input = next(inp for inp in self.cache_lm_session.get_inputs() if inp.name == "cache_key_0")
        self.lm_cache_dtype = self._ort_type_to_numpy(cache_key_input.type)

    @staticmethod
    def _ort_type_to_numpy(ort_type: str) -> np.dtype:
        mapping = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }
        if ort_type not in mapping:
            raise ValueError(f"Unsupported ONNX Runtime tensor type: {ort_type}")
        return mapping[ort_type]

    def _new_cache(self) -> list[np.ndarray]:
        graph = self.cache_graph
        num_layers = int(graph["num_layers"])
        max_total_len = int(graph["max_total_len"])
        num_kv_heads = int(graph["num_key_value_heads"])
        head_dim = int(graph["head_dim"])
        caches: list[np.ndarray] = []
        for _ in range(num_layers):
            caches.extend(
                [
                    np.zeros((1, num_kv_heads, max_total_len, head_dim), dtype=self.lm_cache_dtype),
                    np.zeros((1, num_kv_heads, max_total_len, head_dim), dtype=self.lm_cache_dtype),
                ]
            )
        return caches

    def _run_cache_prefill(self, embeds: np.ndarray, caches: list[np.ndarray]) -> np.ndarray:
        graph = self.cache_graph
        num_layers = int(graph["num_layers"])
        max_total_len = int(graph["max_total_len"])
        prompt_len = int(embeds.shape[0])
        if prompt_len > max_total_len:
            raise ValueError(f"prompt_len exceeds ONNX cache max_total_len: {prompt_len} > {max_total_len}")
        feeds: dict[str, np.ndarray] = {
            "inputs_embeds": embeds[None, :, :].astype(self.lm_embed_dtype, copy=False),
            "cache_position": np.arange(prompt_len, dtype=np.int64),
        }
        outputs = self.prefill_lm_session.run(None, feeds)
        logits = outputs[0][0, -1, :].astype(np.float32, copy=False)
        for i in range(num_layers):
            output_base = 1 + 2 * i
            cache_base = 2 * i
            caches[cache_base][:, :, :prompt_len, :] = outputs[output_base]
            caches[cache_base + 1][:, :, :prompt_len, :] = outputs[output_base + 1]
        return logits

    def _run_cache_token(
        self,
        token_embed: np.ndarray,
        caches: list[np.ndarray],
        *,
        position: int,
        valid_len: int,
    ) -> np.ndarray:
        graph = self.cache_graph
        num_layers = int(graph["num_layers"])
        max_total_len = int(graph["max_total_len"])
        if valid_len > max_total_len:
            raise ValueError(f"valid_len exceeds ONNX cache max_total_len: {valid_len} > {max_total_len}")
        attention_mask = np.zeros((1, max_total_len), dtype=np.int64)
        attention_mask[:, :valid_len] = 1
        feeds: dict[str, np.ndarray] = {
            "inputs_embeds": token_embed.reshape(1, 1, -1).astype(self.lm_embed_dtype, copy=False),
            "attention_mask": attention_mask,
            "cache_position": np.asarray([position], dtype=np.int64),
        }
        for i in range(num_layers):
            base = 2 * i
            feeds[f"cache_key_{i}"] = caches[base]
            feeds[f"cache_value_{i}"] = caches[base + 1]

        outputs = self.cache_lm_session.run(None, feeds)
        logits = outputs[0][0, -1, :].astype(np.float32, copy=False)
        for i in range(num_layers):
            output_base = 1 + 2 * i
            cache_base = 2 * i
            caches[cache_base][:, :, position : position + 1, :] = outputs[output_base]
            caches[cache_base + 1][:, :, position : position + 1, :] = outputs[output_base + 1]
        return logits

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.5,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        hotwords: str | list | None = None,
        hotword_topk: int = 50,
        hotword_start_boost: float = 6.0,
        hotword_continuation_boost: float = 8.0,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        audio = load_audio_bytes(audio_bytes, self.sampling_rate)
        feature, sample_count, encoder_feature_len, hop_length = self._extract_features(audio)
        audio_embeddings = self._audio_embeddings(feature, sample_count, encoder_feature_len, hop_length)
        token_ids, embeds = self._initial_embeddings(audio_embeddings, language=language)
        hotword_list = parse_hotwords(hotwords)
        hot_trie, hot_meta = (
            self._build_hotword_trie(hotword_list, hotword_start_boost, hotword_continuation_boost)
            if hotword_list
            else (None, None)
        )

        max_total_len = int(self.cache_graph["max_total_len"])
        if embeds.shape[0] + int(max_new_tokens) > max_total_len:
            raise ValueError(
                f"Prompt + max_new_tokens exceeds cache max_total_len: "
                f"{embeds.shape[0]} + {max_new_tokens} > {max_total_len}"
            )

        caches = self._new_cache()
        if embeds.shape[0] <= 0:
            raise RuntimeError("Empty prompt")
        logits = self._run_cache_prefill(embeds, caches)

        generated: list[int] = []
        hit_stop = False
        stop_token_id: int | None = None
        rng = np.random.default_rng()
        current_position = embeds.shape[0]
        for _ in range(int(max_new_tokens)):
            step_logits = logits.copy()
            apply_repetition_penalty(step_logits, token_ids + generated, float(repetition_penalty))
            self._mask_logits(step_logits)
            if hot_trie:
                self._apply_hotword_boost(step_logits, generated, hot_trie, hotword_topk)
            if do_sample:
                probs = softmax(step_logits / max(float(temperature), 1e-6))
                next_token = int(rng.choice(np.arange(probs.shape[0]), p=probs))
            else:
                next_token = int(np.argmax(step_logits))
            if next_token in self.eos_token_ids or next_token == self.pad_token_id:
                hit_stop = True
                stop_token_id = next_token
                break
            generated.append(next_token)
            token_embed = self.token_embedding[np.asarray([next_token], dtype=np.int64)][0].astype(np.float32)
            logits = self._run_cache_token(
                token_embed,
                caches,
                position=current_position,
                valid_len=current_position + 1,
            )
            current_position += 1

        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        text = normalize_prediction_text(raw)
        elapsed = time.perf_counter() - started
        return {
            "text": text,
            "raw": raw,
            "generated_tokens": len(generated),
            "hit_stop": hit_stop,
            "stop_token_id": stop_token_id,
            "elapsed_seconds": elapsed,
            "audio_seconds": float(audio.shape[0]) / float(self.sampling_rate),
            "audio_token_count": int(audio_embeddings.shape[0]),
            "providers": self.providers,
            "backend": "onnx_cache",
            "cache_precision": self.cache_precision,
            "audio_precision": self.audio_precision,
            "hotword": hot_meta,
        }


def collect_metrics() -> dict[str, Any]:
    process = psutil.Process()
    vm = psutil.virtual_memory()
    metrics: dict[str, Any] = {
        "process": {
            "pid": process.pid,
            "rss_bytes": int(process.memory_info().rss),
            "cpu_percent": process.cpu_percent(interval=None),
        },
        "system": {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
            "used_bytes": int(vm.used),
            "percent": float(vm.percent),
        },
        "gpu": {
            "nvidia": None,
            "apple": None,
        },
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.5,
        )
        rows = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used_mb": float(parts[2]),
                        "memory_total_mb": float(parts[3]),
                        "utilization_percent": float(parts[4]),
                    }
                )
        metrics["gpu"]["nvidia"] = rows
    except Exception:
        metrics["gpu"]["nvidia"] = []
    return metrics
