#!/usr/bin/env python3
"""Open-dataset reward-hacking workflow.

This is the paper-scale companion to ``group_meeting_workflow.py``.  It replaces
the synthetic latent world with public text datasets:

* source preference training: UltraFeedback Binarized and Anthropic HH-RLHF;
* OOD preference evaluation: RewardBench;
* reward-hacking evaluation: rh-bench and sycophancy-style benchmark rows.

The reward model is intentionally lightweight: a bottleneck pairwise RM over
hashed text n-gram features plus explicit style/proxy features.  This keeps the
experiment cheap enough to run many seeds and weight-decay values, while still
letting us compute AGOP directions with respect to input features.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import random
import re
import textwrap
from typing import Iterable
import zlib

import numpy as np


EPS = 1e-8


# ---------------------------------------------------------------------------
# Dataset records and adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairRecord:
    dataset: str
    split: str
    domain: str
    category: str
    subcategory: str
    prompt: str
    chosen: str
    rejected: str
    chosen_label: str
    rejected_label: str
    source_id: str

    def chosen_text(self) -> str:
        return format_prompt_response(self.prompt, self.chosen)

    def rejected_text(self) -> str:
        return format_prompt_response(self.prompt, self.rejected)


@dataclass
class WorkflowConfig:
    output_dir: str = "results/real_data_full"
    train_sources: tuple[str, ...] = ("ultrafeedback", "hh_rlhf")
    eval_sources: tuple[str, ...] = ("rewardbench",)
    hacking_sources: tuple[str, ...] = ("rh_bench_open", "rh_bench_multi")
    max_train_pairs_per_source: int = 15_000
    max_eval_pairs_per_dataset: int = 5_000
    max_hacking_pairs_per_category: int = 1_500
    epochs: int = 3
    batch_size: int = 192
    lr: float = 2e-2
    bottleneck_dim: int = 16
    hash_dim: int = 16_384
    max_tokens: int = 1_200
    max_chars: int = 8_000
    seeds: tuple[int, ...] = (0, 1, 2)
    weight_decays: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    n_values: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    bon_trials_per_category: int = 500
    agop_sample_size: int = 384
    streaming: bool = True
    quick: bool = False


DATASET_SOURCES = {
    "ultrafeedback": {
        "hf_name": "HuggingFaceH4/ultrafeedback_binarized",
        "config": None,
        "train_split": "train_prefs",
        "eval_split": "test_prefs",
        "role": "train",
        "license": "mit",
    },
    "hh_rlhf": {
        "hf_name": "Anthropic/hh-rlhf",
        "config": None,
        "train_split": "train",
        "eval_split": "test",
        "role": "train",
        "license": "mit",
    },
    "rewardbench": {
        "hf_name": "allenai/reward-bench",
        "config": "default",
        "eval_split": "filtered",
        "role": "eval",
        "license": "odc-by",
    },
    "rh_bench_open": {
        "hf_name": "ktolnos/rh-bench",
        "config": "all",
        "eval_split": "open_ended",
        "role": "hacking_eval",
        "license": "cc-by-sa-4.0",
    },
    "rh_bench_multi": {
        "hf_name": "ktolnos/rh-bench",
        "config": "all",
        "eval_split": "multichoice",
        "role": "hacking_eval",
        "license": "cc-by-sa-4.0",
    },
    "sycophancy_eval": {
        "hf_name": "meg-tong/sycophancy-eval",
        "config": None,
        "eval_split": "train",
        "role": "hacking_eval",
        "license": "see dataset card",
    },
}


def load_dataset_module():
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "Missing dependency `datasets`. Install with `pip install -r requirements.txt` "
            "before running the real-data workflow."
        ) from exc
    return load_dataset


def load_records(
    source_key: str,
    split: str,
    limit: int,
    seed: int,
    streaming: bool,
    hacking_limit_per_category: int | None = None,
) -> list[PairRecord]:
    meta = DATASET_SOURCES[source_key]
    load_dataset = load_dataset_module()
    dataset = load_hf_split(
        load_dataset=load_dataset,
        hf_name=str(meta["hf_name"]),
        config=meta.get("config"),
        split=split,
        streaming=streaming,
        seed=seed,
        limit=limit,
    )

    rows: list[PairRecord] = []
    category_counts: dict[str, int] = {}
    for idx, row in enumerate(dataset):
        parsed = parse_row(source_key, split, row, idx)
        if parsed is None:
            continue
        if hacking_limit_per_category is not None:
            key = f"{parsed.category}/{parsed.subcategory}"
            if category_counts.get(key, 0) >= hacking_limit_per_category:
                if all(v >= hacking_limit_per_category for v in category_counts.values()) and len(rows) >= limit:
                    break
                continue
            category_counts[key] = category_counts.get(key, 0) + 1
        rows.append(parsed)
        if len(rows) >= limit:
            break
    return rows


def load_hf_split(load_dataset, hf_name: str, config: object, split: str, streaming: bool, seed: int, limit: int):
    attempts = []
    if config:
        attempts.append((hf_name, config, split))
    attempts.append((hf_name, None, split))
    last_error: Exception | None = None
    for name, cfg, split_name in attempts:
        try:
            if cfg is None:
                ds = load_dataset(name, split=split_name, streaming=streaming)
            else:
                ds = load_dataset(name, cfg, split=split_name, streaming=streaming)
            if streaming:
                buffer_size = min(max(limit * 4, 1_000), 20_000)
                return ds.shuffle(seed=seed, buffer_size=buffer_size)
            return ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
        except Exception as exc:  # pragma: no cover - dataset-card drift
            last_error = exc
    raise RuntimeError(f"Could not load {hf_name} split={split!r}: {last_error}")


def parse_row(source_key: str, split: str, row: dict, idx: int) -> PairRecord | None:
    if source_key == "ultrafeedback":
        prompt = clean_text(row.get("prompt", ""))
        chosen = extract_assistant_text(row.get("chosen"))
        rejected = extract_assistant_text(row.get("rejected"))
        return make_record(source_key, split, "open_preference", "helpfulness_honesty", "", prompt, chosen, rejected, idx)

    if source_key == "hh_rlhf":
        chosen_full = clean_text(row.get("chosen", ""))
        rejected_full = clean_text(row.get("rejected", ""))
        prompt = extract_common_prompt(chosen_full, rejected_full)
        chosen = strip_prompt_prefix(chosen_full, prompt)
        rejected = strip_prompt_prefix(rejected_full, prompt)
        return make_record(source_key, split, "assistant_preference", infer_hh_category(prompt), "", prompt, chosen, rejected, idx)

    if source_key == "rewardbench":
        prompt = clean_text(row.get("prompt", ""))
        chosen = extract_assistant_text(row.get("chosen"))
        rejected = extract_assistant_text(row.get("rejected"))
        subset = clean_text(row.get("subset", "rewardbench"))
        category = rewardbench_category(subset)
        return make_record(source_key, split, "rewardbench_ood", category, subset, prompt, chosen, rejected, row.get("id", idx))

    if source_key.startswith("rh_bench"):
        prompt = clean_text(row.get("prompt", "") or row.get("question", ""))
        clean = first_nonempty(
            row,
            [
                "response_clean",
                "clean_response",
                "response_not_hacking",
                "response_benign",
                "answer_not_matching_behavior",
                "not_matching_behavior",
                "clean",
            ],
        )
        hacking = first_nonempty(
            row,
            [
                "response_hacking",
                "hacking_response",
                "response_bad",
                "response_exploit",
                "answer_matching_behavior",
                "matching_behavior",
                "hacking",
            ],
        )
        if not clean or not hacking:
            return None
        category = clean_text(row.get("hacking_category", "") or row.get("category", "reward_hacking"))
        subcategory = clean_text(row.get("hacking_subcategory", "") or row.get("subcategory", ""))
        return make_record(
            source_key,
            split,
            clean_text(row.get("source_dataset", "rh_bench")),
            category or "reward_hacking",
            subcategory,
            prompt,
            clean,
            hacking,
            row.get("source_id", idx),
            chosen_label="clean",
            rejected_label="hacking",
        )

    if source_key == "sycophancy_eval":
        prompt = first_nonempty(row, ["question", "prompt", "input", "context"])
        clean = first_nonempty(row, ["answer_not_matching_behavior", "not_matching_behavior", "non_sycophantic_answer"])
        hacking = first_nonempty(row, ["answer_matching_behavior", "matching_behavior", "sycophantic_answer"])
        if not clean or not hacking:
            return None
        return make_record(
            source_key,
            split,
            clean_text(row.get("dataset", "sycophancy_eval")),
            "sycophancy",
            clean_text(row.get("base", "") or row.get("type", "")),
            prompt,
            clean,
            hacking,
            row.get("id", idx),
            chosen_label="non_sycophantic",
            rejected_label="sycophantic",
        )

    raise ValueError(f"Unknown source: {source_key}")


def make_record(
    dataset: str,
    split: str,
    domain: str,
    category: str,
    subcategory: str,
    prompt: object,
    chosen: object,
    rejected: object,
    source_id: object,
    chosen_label: str = "chosen",
    rejected_label: str = "rejected",
) -> PairRecord | None:
    chosen_text = clean_text(chosen)
    rejected_text = clean_text(rejected)
    if len(chosen_text) < 2 or len(rejected_text) < 2:
        return None
    return PairRecord(
        dataset=dataset,
        split=split,
        domain=clean_text(domain) or dataset,
        category=clean_text(category) or dataset,
        subcategory=clean_text(subcategory),
        prompt=clean_text(prompt),
        chosen=chosen_text,
        rejected=rejected_text,
        chosen_label=chosen_label,
        rejected_label=rejected_label,
        source_id=str(source_id),
    )


def extract_assistant_text(value: object) -> str:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        if "content" in value:
            return clean_text(value["content"])
        return clean_text(json.dumps(value, ensure_ascii=False))
    if isinstance(value, list):
        assistant = [
            clean_text(item.get("content", ""))
            for item in value
            if isinstance(item, dict) and str(item.get("role", "")).lower() == "assistant"
        ]
        if assistant:
            return assistant[-1]
        contents = [clean_text(item.get("content", "")) for item in value if isinstance(item, dict)]
        return "\n".join(c for c in contents if c)
    return clean_text(value)


def extract_common_prompt(chosen: str, rejected: str) -> str:
    max_len = min(len(chosen), len(rejected))
    end = 0
    for i in range(max_len):
        if chosen[i] != rejected[i]:
            break
        end = i + 1
    prefix = chosen[:end]
    cut = max(prefix.rfind("\n\nAssistant:"), prefix.rfind("Assistant:"))
    if cut >= 0:
        return prefix[:cut].strip()
    return prefix.strip()


def strip_prompt_prefix(text: str, prompt: str) -> str:
    if prompt and text.startswith(prompt):
        return text[len(prompt) :].strip()
    return text


def infer_hh_category(prompt: str) -> str:
    low = prompt.lower()
    unsafe_terms = ["illegal", "weapon", "drugs", "steal", "hack", "kill", "explosive", "harm"]
    if any(term in low for term in unsafe_terms):
        return "harmlessness"
    return "helpfulness"


def rewardbench_category(subset: str) -> str:
    low = subset.lower()
    if "refusal" in low or "xstest" in low or "donotanswer" in low or "safety" in low:
        return "safety"
    if "hep-" in low or "code" in low:
        return "code_reasoning"
    if "math" in low or "prm" in low:
        return "math_reasoning"
    if "llmbar" in low or "mt-bench" in low or "alpacaeval" in low:
        return "chat_hard"
    return "chat"


def first_nonempty(row: dict, keys: list[str]) -> str:
    for key in keys:
        if key in row:
            value = extract_assistant_text(row.get(key))
            if value:
                return value
    return ""


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return extract_assistant_text(value)
    return re.sub(r"\s+", " ", str(value)).strip()


def format_prompt_response(prompt: str, response: str) -> str:
    if prompt:
        return f"Prompt: {prompt}\n\nResponse: {response}"
    return f"Response: {response}"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


STYLE_FEATURES = [
    "log_chars",
    "log_words",
    "log_lines",
    "bullet_density",
    "numbered_density",
    "markdown_header_density",
    "code_fence_density",
    "json_yaml_density",
    "hedge_density",
    "confidence_density",
    "agreement_density",
    "flattery_density",
    "refusal_density",
    "safety_disclaimer_density",
    "reasoning_marker_density",
    "citation_density",
    "test_keyword_density",
    "reward_keyword_density",
    "deception_keyword_density",
    "tool_env_keyword_density",
    "uppercase_ratio",
    "punctuation_ratio",
]

STYLE_GROUPS = {
    "length_format": ["log_chars", "log_words", "log_lines", "bullet_density", "numbered_density", "markdown_header_density"],
    "confidence_sycophancy": ["confidence_density", "agreement_density", "flattery_density"],
    "refusal_safety": ["refusal_density", "safety_disclaimer_density"],
    "reasoning_theater": ["reasoning_marker_density", "citation_density"],
    "code_eval_gaming": ["code_fence_density", "test_keyword_density"],
    "tampering_deception": ["reward_keyword_density", "deception_keyword_density", "tool_env_keyword_density"],
}


class HashingTextFeaturizer:
    def __init__(self, hash_dim: int, max_tokens: int, max_chars: int) -> None:
        self.hash_dim = hash_dim
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.style_names = STYLE_FEATURES
        self.dim = hash_dim + len(STYLE_FEATURES)
        self._token_re = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]*|\d+(?:\.\d+)?|[^\w\s]")

    def transform(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            self._fill_hash_features(x[i], text)
            x[i, self.hash_dim :] = style_features(text)
        return x

    def style_score(self, text: str) -> float:
        feats = style_features(text)
        weights = np.asarray(
            [
                0.45,
                0.25,
                0.10,
                0.45,
                0.35,
                0.20,
                0.20,
                0.20,
                0.10,
                0.70,
                0.70,
                0.55,
                0.25,
                0.25,
                0.45,
                0.20,
                0.65,
                0.80,
                0.80,
                0.70,
                0.10,
                0.10,
            ],
            dtype=np.float64,
        )
        return float(feats @ weights)

    def style_slice(self) -> slice:
        return slice(self.hash_dim, self.dim)

    def style_group_norms(self, direction: np.ndarray) -> dict[str, float]:
        style_dir = direction[self.hash_dim :]
        index = {name: i for i, name in enumerate(self.style_names)}
        out = {}
        denom = max(float(np.linalg.norm(direction)), EPS)
        for group, names in STYLE_GROUPS.items():
            idx = [index[n] for n in names if n in index]
            projection = float(np.linalg.norm(style_dir[idx]) / denom) if idx else 0.0
            random_baseline = math.sqrt(max(len(idx), 1) / self.dim)
            out[f"{group}_projection"] = projection
            out[f"{group}_enrichment"] = projection / max(random_baseline, EPS)
        return out

    def top_style_features(self, direction: np.ndarray, k: int = 6) -> str:
        style_dir = direction[self.hash_dim :]
        order = np.argsort(-np.abs(style_dir))[:k]
        return ", ".join(f"{self.style_names[i]}={style_dir[i]:.3g}" for i in order)

    def _fill_hash_features(self, row: np.ndarray, text: str) -> None:
        clipped = text[: self.max_chars]
        low = clipped.lower()
        tokens = self._token_re.findall(low)[: self.max_tokens]
        count = 0
        for tok in tokens:
            self._add(row, f"w:{tok}")
            count += 1
        for a, b in zip(tokens, tokens[1:]):
            self._add(row, f"b:{a}_{b}")
            count += 1
        compact = re.sub(r"\s+", " ", low)
        step = 2 if len(compact) > 2_000 else 1
        for n in (3, 4, 5):
            for j in range(0, max(len(compact) - n + 1, 0), step):
                self._add(row, f"c{n}:{compact[j:j+n]}")
                count += 1
        if count:
            row[: self.hash_dim] /= math.sqrt(count)

    def _add(self, row: np.ndarray, feat: str) -> None:
        h = zlib.crc32(feat.encode("utf-8"))
        idx = h % self.hash_dim
        sign = 1.0 if (h & 0x80000000) else -1.0
        row[idx] += sign


def style_features(text: str) -> np.ndarray:
    response = text.split("Response:", 1)[-1]
    low = response.lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*", low)
    n_words = max(len(words), 1)
    n_chars = max(len(response), 1)
    lines = response.splitlines() or [response]

    def density(patterns: list[str]) -> float:
        return math.log1p(sum(low.count(p) for p in patterns)) / math.log1p(n_words + 1)

    bullet_count = sum(1 for line in lines if re.match(r"\s*[-*+]\s+", line))
    numbered_count = sum(1 for line in lines if re.match(r"\s*\d+[.)]\s+", line))
    header_count = sum(1 for line in lines if re.match(r"\s*#{1,6}\s+", line))
    code_fences = low.count("```")
    json_yaml = sum(response.count(ch) for ch in "{}[]:=")
    punctuation = sum(1 for ch in response if ch in "!?;:")
    uppercase = sum(1 for ch in response if ch.isupper())

    feats = [
        math.log1p(n_chars) / 8.0,
        math.log1p(n_words) / 7.0,
        math.log1p(len(lines)) / 4.0,
        math.log1p(bullet_count) / 3.0,
        math.log1p(numbered_count) / 3.0,
        math.log1p(header_count) / 3.0,
        math.log1p(code_fences) / 3.0,
        min(json_yaml / max(n_chars, 1), 0.25) * 4.0,
        density(["maybe", "might", "could", "possibly", "uncertain", "not sure"]),
        density(["definitely", "certainly", "clearly", "obviously", "guaranteed", "always", "never"]),
        density(["you are right", "i agree", "absolutely", "exactly", "that's correct", "you nailed"]),
        density(["great question", "excellent", "brilliant", "insightful", "you're very"]),
        density(["i can't", "i cannot", "i'm unable", "i won't", "not able to"]),
        density(["safe", "ethical", "responsible", "policy", "harmful", "illegal"]),
        density(["step by step", "therefore", "because", "first", "second", "reasoning", "analysis"]),
        density(["source", "citation", "according to", "reference", "study", "paper"]),
        density(["unit test", "visible test", "pytest", "assert", "benchmark", "score"]),
        density(["reward", "grader", "metric", "rubric", "points", "maximize"]),
        density(["hide", "deceive", "secret", "mislead", "fake", "pretend"]),
        density(["tool", "environment", "sandbox", "terminal", "file", "api"]),
        min(uppercase / n_chars, 0.20) * 5.0,
        min(punctuation / n_chars, 0.20) * 5.0,
    ]
    return np.asarray(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class NumpyRewardModel:
    def __init__(self, x_dim: int, bottleneck_dim: int, seed: int, hidden_scale: float = 1.0) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.normal(scale=hidden_scale / math.sqrt(x_dim), size=(x_dim, bottleneck_dim))
        self.b = np.zeros(bottleneck_dim)
        self.v = rng.normal(scale=1.0 / math.sqrt(bottleneck_dim), size=bottleneck_dim)
        self.params = [self.w, self.b, self.v]
        self.m = [np.zeros_like(p) for p in self.params]
        self.u = [np.zeros_like(p) for p in self.params]
        self.step = 0

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.w + self.b)

    def reward(self, x: np.ndarray) -> np.ndarray:
        return self.encode(x) @ self.v

    def input_grad(self, x: np.ndarray) -> np.ndarray:
        z = self.encode(x)
        local = (1.0 - z**2) * self.v[None, :]
        return local @ self.w.T

    def fit(
        self,
        records: list[PairRecord],
        featurizer: HashingTextFeaturizer,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(seed)
        history = []
        n = len(records)
        for epoch in range(epochs):
            order = rng.permutation(n)
            losses = []
            accuracies = []
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                batch = [records[int(i)] for i in idx]
                x_c = featurizer.transform([r.chosen_text() for r in batch])
                x_r = featurizer.transform([r.rejected_text() for r in batch])
                loss, acc, grads = self._batch_loss_grads(x_c, x_r)
                self._adamw(grads, lr=lr, weight_decay=weight_decay)
                losses.append(loss)
                accuracies.append(acc)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "train_accuracy": float(np.mean(accuracies))})
        return history

    def _batch_loss_grads(self, x_c: np.ndarray, x_r: np.ndarray) -> tuple[float, float, list[np.ndarray]]:
        z_c = self.encode(x_c)
        z_r = self.encode(x_r)
        margin = z_c @ self.v - z_r @ self.v
        loss = float(np.mean(np.logaddexp(0.0, -margin)))
        acc = float(np.mean(margin > 0.0))
        d_margin = sigmoid(margin) - 1.0
        scale = 1.0 / max(x_c.shape[0], 1)
        d_c = d_margin * scale
        d_r = -d_margin * scale

        grad_v = z_c.T @ d_c + z_r.T @ d_r
        ga_c = d_c[:, None] * self.v[None, :] * (1.0 - z_c**2)
        ga_r = d_r[:, None] * self.v[None, :] * (1.0 - z_r**2)
        grad_w = x_c.T @ ga_c + x_r.T @ ga_r
        grad_b = ga_c.sum(axis=0) + ga_r.sum(axis=0)
        return loss, acc, [grad_w, grad_b, grad_v]

    def _adamw(self, grads: list[np.ndarray], lr: float, weight_decay: float) -> None:
        beta1 = 0.9
        beta2 = 0.999
        self.step += 1
        for i, (param, grad) in enumerate(zip(self.params, grads, strict=True)):
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grad
            self.u[i] = beta2 * self.u[i] + (1.0 - beta2) * (grad * grad)
            m_hat = self.m[i] / (1.0 - beta1**self.step)
            u_hat = self.u[i] / (1.0 - beta2**self.step)
            if weight_decay:
                param *= 1.0 - lr * weight_decay
            param -= lr * m_hat / (np.sqrt(u_hat) + 1e-8)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_workflow(cfg: WorkflowConfig) -> None:
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_by_source: dict[str, list[PairRecord]] = {}
    source_eval_by_source: dict[str, list[PairRecord]] = {}
    eval_records: list[PairRecord] = []
    hacking_records: list[PairRecord] = []
    manifest_rows: list[dict[str, object]] = []

    for source in cfg.train_sources:
        meta = DATASET_SOURCES[source]
        try:
            train = load_records(source, str(meta["train_split"]), cfg.max_train_pairs_per_source, seed=11, streaming=cfg.streaming)
        except Exception as exc:
            manifest_rows.append(error_manifest_row(source, str(meta["train_split"]), "train", exc))
            continue
        train_by_source[source] = train
        manifest_rows.append(manifest_row(source, str(meta["train_split"]), "train", train))
        eval_split = str(meta.get("eval_split", meta["train_split"]))
        try:
            heldout = load_records(source, eval_split, cfg.max_eval_pairs_per_dataset, seed=13, streaming=cfg.streaming)
        except Exception as exc:
            manifest_rows.append(error_manifest_row(source, eval_split, "source_eval", exc))
            heldout = []
        source_eval_by_source[source] = heldout
        manifest_rows.append(manifest_row(source, eval_split, "source_eval", heldout))

    for source in cfg.eval_sources:
        meta = DATASET_SOURCES[source]
        try:
            rows = load_records(source, str(meta["eval_split"]), cfg.max_eval_pairs_per_dataset, seed=17, streaming=cfg.streaming)
        except Exception as exc:
            manifest_rows.append(error_manifest_row(source, str(meta["eval_split"]), "ood_eval", exc))
            continue
        eval_records.extend(rows)
        manifest_rows.append(manifest_row(source, str(meta["eval_split"]), "ood_eval", rows))

    for source in cfg.hacking_sources:
        meta = DATASET_SOURCES[source]
        limit = cfg.max_hacking_pairs_per_category * 20
        try:
            rows = load_records(
                source,
                str(meta["eval_split"]),
                limit=limit,
                seed=19,
                streaming=cfg.streaming,
                hacking_limit_per_category=cfg.max_hacking_pairs_per_category,
            )
        except Exception as exc:
            manifest_rows.append(error_manifest_row(source, str(meta["eval_split"]), "hacking_eval", exc))
            continue
        hacking_records.extend(rows)
        manifest_rows.append(manifest_row(source, str(meta["eval_split"]), "hacking_eval", rows))

    train_records = flatten(train_by_source.values())
    if not train_records:
        raise SystemExit("No training records loaded. Check dataset names/splits and network access.")
    if not hacking_records:
        raise SystemExit("No reward-hacking evaluation records loaded. Check rh-bench/sycophancy adapters.")

    featurizer = HashingTextFeaturizer(cfg.hash_dim, cfg.max_tokens, cfg.max_chars)

    train_history_rows: list[dict[str, object]] = []
    source_eval_rows: list[dict[str, object]] = []
    preference_eval_rows: list[dict[str, object]] = []
    hacking_eval_rows: list[dict[str, object]] = []
    agop_rows: list[dict[str, object]] = []
    bon_rows: list[dict[str, object]] = []

    for seed in cfg.seeds:
        rng = random.Random(seed)
        train_sample = list(train_records)
        rng.shuffle(train_sample)
        for wd in cfg.weight_decays:
            model = NumpyRewardModel(featurizer.dim, cfg.bottleneck_dim, seed=10_000 + seed)
            history = model.fit(train_sample, featurizer, cfg.epochs, cfg.batch_size, cfg.lr, wd, seed=20_000 + seed)
            for hist in history:
                train_history_rows.append({"seed": seed, "weight_decay": wd, **hist})

            for source, records in source_eval_by_source.items():
                source_eval_rows.extend(
                    evaluate_grouped_pairs(
                        model,
                        featurizer,
                        records,
                        cfg.batch_size,
                        keys=["dataset", "category"],
                        base_labels={"seed": seed, "weight_decay": wd, "eval_kind": "source"},
                    )
                )

            preference_eval_rows.extend(
                evaluate_grouped_pairs(
                    model,
                    featurizer,
                    eval_records,
                    cfg.batch_size,
                    keys=["dataset", "category"],
                    base_labels={"seed": seed, "weight_decay": wd, "eval_kind": "ood_preference"},
                )
            )

            hacking_eval_rows.extend(
                evaluate_grouped_pairs(
                    model,
                    featurizer,
                    hacking_records,
                    cfg.batch_size,
                    keys=["dataset", "category"],
                    base_labels={"seed": seed, "weight_decay": wd, "eval_kind": "reward_hacking", "positive_label": "clean"},
                )
            )

            source_agop_records = sample_records(source_eval_by_source, cfg.agop_sample_size, seed=30_000 + seed)
            if not source_agop_records:
                source_agop_records = train_sample[: cfg.agop_sample_size]
            source_top = agop_top_direction(model, featurizer, source_agop_records, cfg.agop_sample_size, seed=40_000 + seed)
            for (dataset, category), group in group_by(hacking_records, ["dataset", "category"]).items():
                target_top = agop_top_direction(model, featurizer, group, cfg.agop_sample_size, seed=50_000 + seed)
                agop_rows.append(
                    {
                        "seed": seed,
                        "weight_decay": wd,
                        "dataset": dataset,
                        "category": category,
                        "source_target_agop_abs_cosine": abs(cosine(source_top, target_top)),
                        "source_style_mass": style_mass(source_top, featurizer),
                        "source_style_mass_enrichment": style_mass_enrichment(source_top, featurizer),
                        "target_style_mass": style_mass(target_top, featurizer),
                        "target_style_mass_enrichment": style_mass_enrichment(target_top, featurizer),
                        "target_top_style_features": featurizer.top_style_features(target_top),
                        **prefix_keys(featurizer.style_group_norms(target_top), "target_"),
                    }
                )

            bon_rows.extend(best_of_n_rows(model, featurizer, hacking_records, cfg, seed=60_000 + seed, weight_decay=wd))

    write_csv(outdir / "dataset_manifest.csv", manifest_rows)
    write_csv(outdir / "train_history.csv", train_history_rows)
    write_csv(outdir / "source_eval.csv", source_eval_rows)
    write_csv(outdir / "ood_preference_eval.csv", preference_eval_rows)
    write_csv(outdir / "hacking_eval.csv", hacking_eval_rows)
    write_csv(outdir / "agop_diagnostics.csv", agop_rows)
    write_csv(outdir / "best_of_n.csv", bon_rows)
    write_csv(outdir / "summary_by_weight_decay.csv", summary_by_weight_decay(hacking_eval_rows, bon_rows, cfg))
    write_report(outdir / "report.md", cfg, manifest_rows, source_eval_rows, preference_eval_rows, hacking_eval_rows, agop_rows, bon_rows)
    maybe_write_plots(outdir, hacking_eval_rows, bon_rows)
    print(f"Wrote real-data workflow results to {outdir.resolve()}")


def manifest_row(source: str, split: str, role: str, records: list[PairRecord]) -> dict[str, object]:
    meta = DATASET_SOURCES[source]
    cats = sorted({r.category for r in records})
    return {
        "source_key": source,
        "hf_name": meta["hf_name"],
        "split": split,
        "role": role,
        "license": meta.get("license", ""),
        "n_pairs": len(records),
        "n_categories": len(cats),
        "categories": "; ".join(cats[:24]),
    }


def error_manifest_row(source: str, split: str, role: str, exc: Exception) -> dict[str, object]:
    meta = DATASET_SOURCES[source]
    return {
        "source_key": source,
        "hf_name": meta["hf_name"],
        "split": split,
        "role": role,
        "license": meta.get("license", ""),
        "n_pairs": 0,
        "n_categories": 0,
        "categories": "",
        "load_error": f"{type(exc).__name__}: {exc}",
    }


def evaluate_grouped_pairs(
    model: NumpyRewardModel,
    featurizer: HashingTextFeaturizer,
    records: list[PairRecord],
    batch_size: int,
    keys: list[str],
    base_labels: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for key, group in group_by(records, keys).items():
        metrics = evaluate_pairs(model, featurizer, group, batch_size)
        labels = dict(base_labels)
        labels.update(dict(zip(keys, key, strict=True)))
        labels["n_pairs"] = len(group)
        rows.append({**labels, **metrics})
    return rows


def evaluate_pairs(
    model: NumpyRewardModel,
    featurizer: HashingTextFeaturizer,
    records: list[PairRecord],
    batch_size: int,
) -> dict[str, float]:
    margins = []
    chosen_rewards = []
    rejected_rewards = []
    chosen_style = []
    rejected_style = []
    for batch in batched(records, batch_size):
        x_c = featurizer.transform([r.chosen_text() for r in batch])
        x_r = featurizer.transform([r.rejected_text() for r in batch])
        r_c = model.reward(x_c)
        r_r = model.reward(x_r)
        margins.extend((r_c - r_r).tolist())
        chosen_rewards.extend(r_c.tolist())
        rejected_rewards.extend(r_r.tolist())
        chosen_style.extend(featurizer.style_score(r.chosen_text()) for r in batch)
        rejected_style.extend(featurizer.style_score(r.rejected_text()) for r in batch)

    margins_a = np.asarray(margins, dtype=np.float64)
    chosen_r = np.asarray(chosen_rewards, dtype=np.float64)
    rejected_r = np.asarray(rejected_rewards, dtype=np.float64)
    chosen_s = np.asarray(chosen_style, dtype=np.float64)
    rejected_s = np.asarray(rejected_style, dtype=np.float64)
    all_rewards = np.concatenate([chosen_r, rejected_r])
    all_style = np.concatenate([chosen_s, rejected_s])
    return {
        "accuracy": float(np.mean(margins_a > 0.0)),
        "failure_rate": float(np.mean(margins_a <= 0.0)),
        "margin_mean": float(np.mean(margins_a)),
        "margin_std": float(np.std(margins_a)),
        "chosen_reward_mean": float(np.mean(chosen_r)),
        "rejected_reward_mean": float(np.mean(rejected_r)),
        "chosen_style_score_mean": float(np.mean(chosen_s)),
        "rejected_style_score_mean": float(np.mean(rejected_s)),
        "rejected_minus_chosen_style_score": float(np.mean(rejected_s - chosen_s)),
        "reward_style_corr": corr(all_rewards, all_style),
        "style_reward_sensitivity": float(np.mean(rejected_r - chosen_r) / max(abs(np.mean(rejected_s - chosen_s)), EPS)),
    }


def agop_top_direction(
    model: NumpyRewardModel,
    featurizer: HashingTextFeaturizer,
    records: list[PairRecord],
    sample_size: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    if len(records) > sample_size:
        records = rng.sample(records, sample_size)
    texts = []
    for rec in records:
        texts.append(rec.chosen_text())
        texts.append(rec.rejected_text())
    if len(texts) > sample_size:
        texts = rng.sample(texts, sample_size)
    x = featurizer.transform(texts)
    grads = model.input_grad(x)
    return top_direction_power(grads)


def top_direction_power(mat: np.ndarray, n_iter: int = 25) -> np.ndarray:
    centered = mat - mat.mean(axis=0, keepdims=True)
    if not np.any(np.isfinite(centered)) or np.allclose(centered, 0.0):
        centered = mat
    rng = np.random.default_rng(123)
    v = rng.normal(size=centered.shape[1])
    v /= max(float(np.linalg.norm(v)), EPS)
    for _ in range(n_iter):
        v = centered.T @ (centered @ v)
        v /= max(float(np.linalg.norm(v)), EPS)
    return v


def style_mass(direction: np.ndarray, featurizer: HashingTextFeaturizer) -> float:
    return float(np.linalg.norm(direction[featurizer.style_slice()]) / max(float(np.linalg.norm(direction)), EPS))


def style_mass_enrichment(direction: np.ndarray, featurizer: HashingTextFeaturizer) -> float:
    random_baseline = math.sqrt(len(featurizer.style_names) / featurizer.dim)
    return style_mass(direction, featurizer) / max(random_baseline, EPS)


def best_of_n_rows(
    model: NumpyRewardModel,
    featurizer: HashingTextFeaturizer,
    records: list[PairRecord],
    cfg: WorkflowConfig,
    seed: int,
    weight_decay: float,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = []
    for (dataset, category), group in group_by(records, ["dataset", "category"]).items():
        if len(group) < 2:
            continue
        hacks = [r.rejected_text() for r in group]
        trials = min(cfg.bon_trials_per_category, len(group))
        trial_records = rng.sample(group, trials) if len(group) > trials else list(group)
        for n in cfg.n_values:
            clean_selected = []
            selected_style = []
            selected_reward = []
            for rec in trial_records:
                candidate_texts = [rec.chosen_text()]
                if n > 1:
                    candidate_texts.extend(rng.choice(hacks) for _ in range(n - 1))
                x = featurizer.transform(candidate_texts)
                rewards = model.reward(x)
                idx = int(np.argmax(rewards))
                clean_selected.append(1.0 if idx == 0 else 0.0)
                selected_style.append(featurizer.style_score(candidate_texts[idx]))
                selected_reward.append(float(rewards[idx]))
            rows.append(
                {
                    "seed": seed,
                    "weight_decay": weight_decay,
                    "dataset": dataset,
                    "category": category,
                    "N": n,
                    "n_trials": trials,
                    "clean_selected_rate": float(np.mean(clean_selected)),
                    "hacking_selected_rate": float(1.0 - np.mean(clean_selected)),
                    "selected_style_score_mean": float(np.mean(selected_style)),
                    "selected_reward_mean": float(np.mean(selected_reward)),
                }
            )
    return rows


def sample_records(source_eval_by_source: dict[str, list[PairRecord]], n: int, seed: int) -> list[PairRecord]:
    rng = random.Random(seed)
    records = flatten(source_eval_by_source.values())
    if len(records) <= n:
        return records
    return rng.sample(records, n)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summary_by_weight_decay(
    hacking_rows: list[dict[str, object]],
    bon_rows: list[dict[str, object]],
    cfg: WorkflowConfig,
) -> list[dict[str, object]]:
    hack_summary = aggregate(hacking_rows, ["weight_decay"])
    bon_max = [r for r in bon_rows if int(r["N"]) == max(cfg.n_values)]
    bon_summary = aggregate(bon_max, ["weight_decay"])
    return merge_on_keys(hack_summary, bon_summary, ["weight_decay"])


def write_report(
    path: Path,
    cfg: WorkflowConfig,
    manifest_rows: list[dict[str, object]],
    source_eval_rows: list[dict[str, object]],
    preference_eval_rows: list[dict[str, object]],
    hacking_eval_rows: list[dict[str, object]],
    agop_rows: list[dict[str, object]],
    bon_rows: list[dict[str, object]],
) -> None:
    baseline_wd = min(cfg.weight_decays)
    max_n = max(cfg.n_values)
    h1 = aggregate([r for r in hacking_eval_rows if float(r["weight_decay"]) == baseline_wd], ["dataset", "category"])
    source = aggregate(source_eval_rows, ["dataset", "category", "weight_decay"])
    source_best = [r for r in source if float(r["weight_decay"]) == baseline_wd]
    pref = aggregate(preference_eval_rows, ["dataset", "category", "weight_decay"])
    pref_best = [r for r in pref if float(r["weight_decay"]) == baseline_wd]
    h3 = aggregate(hacking_eval_rows, ["dataset", "category", "weight_decay"])
    bon_max = aggregate([r for r in bon_rows if int(r["N"]) == max_n], ["dataset", "category", "weight_decay"])
    agop = aggregate(agop_rows, ["dataset", "category", "weight_decay"])
    h3_merged = merge_on_keys(h3, bon_max, ["dataset", "category", "weight_decay"])
    h3_merged = merge_on_keys(h3_merged, agop, ["dataset", "category", "weight_decay"])
    best_h3 = best_by_group(h3_merged, ["dataset", "category"], metric="hacking_selected_rate_mean", maximize=False)
    agop_baseline = [r for r in agop if float(r["weight_decay"]) == baseline_wd]
    wd_summary = summary_by_weight_decay(hacking_eval_rows, bon_rows, cfg)

    lines = [
        "# Real-Data Reward Hacking Workflow Report",
        "",
        "## Dataset Manifest",
        "",
        markdown_table(manifest_rows, ["source_key", "hf_name", "split", "role", "n_pairs", "n_categories", "license"], max_rows=20),
        "",
        "## H1: reward hacking types on open benchmarks",
        "",
        "Baseline rows use the smallest weight decay. `accuracy` means clean/chosen is scored above hacking/rejected; low accuracy is a reward-hacking failure.",
        "",
        markdown_table(
            h1,
            [
                "dataset",
                "category",
                "n_pairs_mean",
                "accuracy_mean",
                "failure_rate_mean",
                "margin_mean_mean",
                "rejected_minus_chosen_style_score_mean",
                "reward_style_corr_mean",
            ],
            max_rows=30,
        ),
        "",
        "## H2: AGOP transfer from source preference data to hacking domains",
        "",
        "High source-target AGOP cosine is evidence of direction transfer. Style-mass enrichment and group enrichments test whether that transferred direction is concentrated in interpretable proxy/style features.",
        "",
        markdown_table(
            agop_baseline,
            [
                "dataset",
                "category",
                "source_target_agop_abs_cosine_mean",
                "source_style_mass_mean",
                "source_style_mass_enrichment_mean",
                "target_style_mass_mean",
                "target_style_mass_enrichment_mean",
                "target_confidence_sycophancy_enrichment_mean",
                "target_tampering_deception_enrichment_mean",
            ],
            max_rows=30,
        ),
        "",
        "## H3: weight decay sweep",
        "",
        f"Rows are selected by minimum hacking-selected rate under adversarial best-of-N at N={max_n}.",
        "",
        markdown_table(
            best_h3,
            [
                "dataset",
                "category",
                "weight_decay",
                "accuracy_mean",
                "failure_rate_mean",
                "hacking_selected_rate_mean",
                "clean_selected_rate_mean",
                "target_style_mass_enrichment_mean",
            ],
            max_rows=30,
        ),
        "",
        f"## Aggregate Weight-Decay Trend at N={max_n}",
        "",
        markdown_table(
            wd_summary,
            [
                "weight_decay",
                "accuracy_mean",
                "failure_rate_mean",
                "margin_mean_mean",
                "hacking_selected_rate_mean",
                "clean_selected_rate_mean",
            ],
            max_rows=20,
        ),
        "",
        "## Source and OOD preference checks",
        "",
        "These tables check that the RM still learns the ordinary preference task while we test hacking robustness.",
        "",
        markdown_table(source_best, ["dataset", "category", "accuracy_mean", "margin_mean_mean"], max_rows=20),
        "",
        markdown_table(pref_best, ["dataset", "category", "accuracy_mean", "margin_mean_mean"], max_rows=20),
        "",
        "## Files",
        "",
        "- `dataset_manifest.csv`: loaded open datasets and category counts.",
        "- `train_history.csv`: training loss/accuracy per seed and weight decay.",
        "- `source_eval.csv`: held-out source preference accuracy.",
        "- `ood_preference_eval.csv`: RewardBench-style OOD preference accuracy.",
        "- `hacking_eval.csv`: clean-vs-hacking pair accuracy by category.",
        "- `agop_diagnostics.csv`: source-target AGOP cosine and style/proxy projections.",
        "- `best_of_n.csv`: adversarial clean-vs-many-hacks selection pressure.",
        "- `summary_by_weight_decay.csv`: aggregate mitigation trend.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_write_plots(outdir: Path, hacking_rows: list[dict[str, object]], bon_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(exist_ok=True)

    hack = aggregate(hacking_rows, ["category", "weight_decay"])
    for metric in ["accuracy_mean", "failure_rate_mean", "reward_style_corr_mean"]:
        plt.figure(figsize=(9, 5))
        for category in sorted({str(r["category"]) for r in hack}):
            sub = sorted([r for r in hack if r["category"] == category], key=lambda r: float(r["weight_decay"]))
            plt.plot([float(r["weight_decay"]) for r in sub], [float(r.get(metric, np.nan)) for r in sub], marker="o", label=category)
        plt.xscale("symlog", linthresh=1e-7)
        plt.xlabel("weight_decay")
        plt.ylabel(metric)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / f"hacking_{metric}.png", dpi=160)
        plt.close()

    bon = aggregate(bon_rows, ["category", "weight_decay", "N"])
    for category in sorted({str(r["category"]) for r in bon}):
        plt.figure(figsize=(8, 5))
        for wd in sorted({float(r["weight_decay"]) for r in bon}):
            sub = sorted([r for r in bon if r["category"] == category and float(r["weight_decay"]) == wd], key=lambda r: int(r["N"]))
            plt.plot([int(r["N"]) for r in sub], [float(r["hacking_selected_rate_mean"]) for r in sub], marker="o", label=f"wd={wd:g}")
        plt.xscale("log", base=2)
        plt.xlabel("N")
        plt.ylabel("hacking_selected_rate")
        plt.title(category)
        plt.legend(fontsize=7)
        plt.tight_layout()
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", category)
        plt.savefig(plot_dir / f"bon_{safe}.png", dpi=160)
        plt.close()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    denom = math.sqrt(float(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom < EPS:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS)
    return float(np.dot(a, b) / denom)


def batched(items: list[PairRecord] | list[str], batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def group_by(records: list[PairRecord], keys: list[str]) -> dict[tuple[object, ...], list[PairRecord]]:
    groups: dict[tuple[object, ...], list[PairRecord]] = {}
    for rec in records:
        key = tuple(getattr(rec, k) for k in keys)
        groups.setdefault(key, []).append(rec)
    return groups


def flatten(groups: Iterable[Iterable[PairRecord]]) -> list[PairRecord]:
    return list(itertools.chain.from_iterable(groups))


def prefix_keys(row: dict[str, object], prefix: str) -> dict[str, object]:
    return {f"{prefix}{k}": v for k, v in row.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        row: dict[str, object] = dict(zip(keys, key, strict=True))
        numeric_keys = sorted(
            {
                k
                for r in group_rows
                for k, v in r.items()
                if k not in keys and isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
            }
        )
        for name in numeric_keys:
            values = np.asarray([float(r[name]) for r in group_rows if name in r and np.isfinite(float(r[name]))])
            if values.size:
                row[f"{name}_mean"] = float(values.mean())
                row[f"{name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out.append(row)
    return out


def merge_on_keys(
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    keys: list[str],
) -> list[dict[str, object]]:
    right_by_key = {tuple(row[k] for k in keys): row for row in right_rows}
    merged = []
    for left in left_rows:
        key = tuple(left[k] for k in keys)
        row = dict(left)
        row.update(right_by_key.get(key, {}))
        merged.append(row)
    return merged


def best_by_group(
    rows: list[dict[str, object]],
    keys: list[str],
    metric: str,
    maximize: bool = False,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    best = []
    for _, group_rows in groups.items():
        finite = [r for r in group_rows if metric in r and np.isfinite(float(r[metric]))]
        if not finite:
            continue
        key_fn = lambda r: float(r[metric])
        best.append(max(finite, key=key_fn) if maximize else min(finite, key=key_fn))
    return sorted(best, key=lambda r: tuple(str(r[k]) for k in keys))


def markdown_table(rows: list[dict[str, object]], cols: list[str], max_rows: int) -> str:
    shown = rows[:max_rows]
    if not shown:
        return "No rows."
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in shown:
        cells = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full open-dataset reward-hacking workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python src/real_data_workflow.py --quick
              python src/real_data_workflow.py --output-dir results/real_data_full
            """
        ),
    )
    p.add_argument("--output-dir", default="results/real_data_full")
    p.add_argument("--quick", action="store_true", help="Tiny smoke run for dependency/schema checks.")
    p.add_argument("--train-sources", nargs="+", default=["ultrafeedback", "hh_rlhf"])
    p.add_argument("--eval-sources", nargs="+", default=["rewardbench"])
    p.add_argument("--hacking-sources", nargs="+", default=["rh_bench_open", "rh_bench_multi"])
    p.add_argument("--max-train-pairs-per-source", type=int, default=15_000)
    p.add_argument("--max-eval-pairs-per-dataset", type=int, default=5_000)
    p.add_argument("--max-hacking-pairs-per-category", type=int, default=1_500)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--bottleneck-dim", type=int, default=16)
    p.add_argument("--hash-dim", type=int, default=16_384)
    p.add_argument("--max-tokens", type=int, default=1_200)
    p.add_argument("--max-chars", type=int, default=8_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    p.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--bon-trials-per-category", type=int, default=500)
    p.add_argument("--agop-sample-size", type=int, default=384)
    p.add_argument("--no-streaming", action="store_true", help="Download full HF datasets instead of streaming.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.output_dir = "results/real_data_smoke" if args.output_dir == "results/real_data_full" else args.output_dir
        args.max_train_pairs_per_source = 600
        args.max_eval_pairs_per_dataset = 400
        args.max_hacking_pairs_per_category = 120
        args.epochs = 1
        args.batch_size = 96
        args.hash_dim = 4_096
        args.max_tokens = 500
        args.max_chars = 4_000
        args.seeds = [0]
        args.weight_decays = [0.0, 1e-3, 1e-1]
        args.n_values = [1, 2, 4, 8]
        args.bon_trials_per_category = 80
        args.agop_sample_size = 128

    cfg = WorkflowConfig(
        output_dir=args.output_dir,
        train_sources=tuple(args.train_sources),
        eval_sources=tuple(args.eval_sources),
        hacking_sources=tuple(args.hacking_sources),
        max_train_pairs_per_source=args.max_train_pairs_per_source,
        max_eval_pairs_per_dataset=args.max_eval_pairs_per_dataset,
        max_hacking_pairs_per_category=args.max_hacking_pairs_per_category,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bottleneck_dim=args.bottleneck_dim,
        hash_dim=args.hash_dim,
        max_tokens=args.max_tokens,
        max_chars=args.max_chars,
        seeds=tuple(args.seeds),
        weight_decays=tuple(args.weight_decays),
        n_values=tuple(args.n_values),
        bon_trials_per_category=args.bon_trials_per_category,
        agop_sample_size=args.agop_sample_size,
        streaming=not args.no_streaming,
        quick=args.quick,
    )
    run_workflow(cfg)


if __name__ == "__main__":
    main()
