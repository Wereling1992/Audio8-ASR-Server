"""Backend-agnostic hotword trie logit-boosting core."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Sequence, Set


_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
_BARE_TAG_RE = re.compile(r"</?[^>\s]+>")
_CJK_KANA_HANGUL_RE = re.compile("[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


class HotwordTrie:
    """Prefix trie over hotword token sequences with per-step boost lookup."""

    def __init__(
        self,
        token_sequences: Sequence[Sequence[int]],
        *,
        start_boost: float,
        continuation_boost: float,
    ) -> None:
        self.start_boost = float(start_boost)
        self.continuation_boost = float(continuation_boost)
        self.trie: Dict[int, Dict[int, Any]] = {}
        self.max_sequence_len = 0
        for seq in token_sequences:
            ids = [int(token_id) for token_id in seq]
            if not ids:
                continue
            node = self.trie
            for token_id in ids:
                node = node.setdefault(token_id, {})
            self.max_sequence_len = max(self.max_sequence_len, len(ids))
        self.start_token_ids = sorted(self.trie.keys())

    def __bool__(self) -> bool:
        return bool(self.trie)

    def boosts_for_generated(self, generated_ids: Sequence[int]) -> Dict[int, float]:
        boosts: Dict[int, float] = {}
        if self.start_boost:
            for token_id in self.start_token_ids:
                boosts[token_id] = max(boosts.get(token_id, 0.0), self.start_boost)

        if not generated_ids or not self.continuation_boost or self.max_sequence_len <= 1:
            return boosts

        max_prefix_len = min(len(generated_ids), self.max_sequence_len - 1)
        for prefix_len in range(1, max_prefix_len + 1):
            node: Dict[int, Any] = self.trie
            matched = True
            for token_id in generated_ids[-prefix_len:]:
                next_node = node.get(int(token_id))
                if next_node is None:
                    matched = False
                    break
                node = next_node
            if not matched:
                continue
            for next_token_id in node.keys():
                boosts[int(next_token_id)] = max(
                    boosts.get(int(next_token_id), 0.0), self.continuation_boost
                )
        return boosts


def _has_cjk_or_kana_or_hangul(text: str) -> bool:
    return bool(_CJK_KANA_HANGUL_RE.search(str(text or "")))


def _hotword_text_variants(word: str) -> List[str]:
    word = str(word or "").strip()
    if not word:
        return []
    variants = [word]
    if not _has_cjk_or_kana_or_hangul(word) and re.search(r"[A-Za-z0-9_]", word):
        variants.append(" " + word)
    out: List[str] = []
    seen: Set[str] = set()
    for value in variants:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def token_is_control_or_special(token: str, token_id: int, special_ids: Set[int]) -> bool:
    if int(token_id) in special_ids:
        return True
    token = str(token)
    return bool(_CONTROL_TOKEN_RE.fullmatch(token) or _BARE_TAG_RE.fullmatch(token))


def build_hotword_sequences(
    hotwords: Sequence[str],
    *,
    encode: Callable[[str], List[int]],
    id_to_token: Callable[[int], str],
    special_ids: Set[int],
) -> Dict[str, List[List[int]]]:
    special_ids = set(int(x) for x in special_ids if x is not None)
    sequences: Dict[str, List[List[int]]] = {}
    seen_global: Set[tuple[int, ...]] = set()
    for word in hotwords:
        word = str(word or "").strip()
        if not word:
            continue
        variants: List[List[int]] = []
        for text in _hotword_text_variants(word):
            ids = [
                int(token_id)
                for token_id in encode(text)
                if not token_is_control_or_special(id_to_token(int(token_id)), int(token_id), special_ids)
            ]
            key = tuple(ids)
            if not key or key in seen_global:
                continue
            seen_global.add(key)
            variants.append(ids)
        if variants:
            sequences[word] = variants
    return sequences


def flatten_sequences(sequences_by_word: Dict[str, List[List[int]]]) -> List[List[int]]:
    return [ids for variants in sequences_by_word.values() for ids in variants]


def parse_hotwords(raw: Any) -> List[str]:
    values: List[str] = []
    if isinstance(raw, (list, tuple)):
        values = [str(x).strip() for x in raw]
    elif raw:
        values = [x.strip() for x in re.split(r"[,，]", str(raw))]
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_trie_from_hotwords(
    hotwords: Sequence[str],
    *,
    encode: Callable[[str], List[int]],
    id_to_token: Callable[[int], str],
    special_ids: Set[int],
    start_boost: float,
    continuation_boost: float,
) -> tuple[HotwordTrie, Dict[str, List[List[int]]]]:
    sequences_by_word = build_hotword_sequences(
        hotwords, encode=encode, id_to_token=id_to_token, special_ids=special_ids
    )
    trie = HotwordTrie(
        flatten_sequences(sequences_by_word),
        start_boost=start_boost,
        continuation_boost=continuation_boost,
    )
    return trie, sequences_by_word
