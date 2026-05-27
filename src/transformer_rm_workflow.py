#!/usr/bin/env python3
"""GPU transformer reward-model workflow for open reward-hacking data.

Common reward models are usually a pretrained Transformer backbone plus a
scalar reward head.  They are trained on preference pairs with a Bradley-Terry
loss:

    loss = -log sigmoid(r(chosen) - r(rejected))

This script implements that standard architecture with
``AutoModelForSequenceClassification(num_labels=1)`` so it can run with encoder
models such as ``distilroberta-base``, ``roberta-base``, or
``microsoft/deberta-v3-small``.  It shares dataset adapters with
``real_data_workflow.py`` and adds CUDA training, best-of-N reward-hacking
stress tests, and AGOP diagnostics with respect to token embeddings.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import textwrap
from typing import Any, Iterable

import numpy as np

from real_data_workflow import (
    DATASET_SOURCES,
    HashingTextFeaturizer,
    PairRecord,
    aggregate,
    best_by_group,
    corr,
    cosine,
    error_manifest_row,
    flatten,
    group_by,
    load_records,
    manifest_row,
    markdown_table,
    merge_on_keys,
    write_csv,
)


EPS = 1e-8


@dataclass
class TransformerRMConfig:
    output_dir: str = "results/transformer_rm_full"
    model_name: str = "distilroberta-base"
    train_sources: tuple[str, ...] = ("ultrafeedback", "hh_rlhf")
    eval_sources: tuple[str, ...] = ("rewardbench",)
    hacking_sources: tuple[str, ...] = ("rh_bench_open", "rh_bench_multi")
    max_train_pairs_per_source: int = 12_000
    max_eval_pairs_per_dataset: int = 4_000
    max_hacking_pairs_per_category: int = 1_200
    epochs: int = 2
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    lr: float = 2e-5
    warmup_ratio: float = 0.03
    max_length: int = 512
    seeds: tuple[int, ...] = (0, 1, 2)
    weight_decays: tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
    n_values: tuple[int, ...] = (1, 2, 4, 8, 16)
    bon_trials_per_category: int = 400
    agop_sample_size: int = 128
    device: str = "auto"
    precision: str = "auto"
    freeze_backbone: bool = False
    streaming: bool = True
    quick: bool = False


def require_torch_transformers():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "Missing GPU workflow dependencies. Install with `pip install -r requirements.txt` "
            "and make sure your PyTorch build matches the server CUDA version."
        ) from exc
    return torch, F, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(torch, requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_precision(torch, device, requested: str) -> str:
    if requested != "auto":
        return requested
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    if device.type == "cuda":
        return "fp16"
    return "fp32"


def autocast_context(torch, device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_all_records(cfg: TransformerRMConfig) -> tuple[
    dict[str, list[PairRecord]],
    dict[str, list[PairRecord]],
    list[PairRecord],
    list[PairRecord],
    list[dict[str, object]],
]:
    train_by_source: dict[str, list[PairRecord]] = {}
    source_eval_by_source: dict[str, list[PairRecord]] = {}
    eval_records: list[PairRecord] = []
    hacking_records: list[PairRecord] = []
    manifest_rows: list[dict[str, object]] = []

    for source in cfg.train_sources:
        meta = DATASET_SOURCES[source]
        try:
            train = load_records(
                source,
                str(meta["train_split"]),
                cfg.max_train_pairs_per_source,
                seed=11,
                streaming=cfg.streaming,
            )
        except Exception as exc:
            manifest_rows.append(error_manifest_row(source, str(meta["train_split"]), "train", exc))
            continue
        train_by_source[source] = train
        manifest_rows.append(manifest_row(source, str(meta["train_split"]), "train", train))

        eval_split = str(meta.get("eval_split", meta["train_split"]))
        try:
            heldout = load_records(
                source,
                eval_split,
                cfg.max_eval_pairs_per_dataset,
                seed=13,
                streaming=cfg.streaming,
            )
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
        try:
            rows = load_records(
                source,
                str(meta["eval_split"]),
                limit=cfg.max_hacking_pairs_per_category * 20,
                seed=19,
                streaming=cfg.streaming,
                hacking_limit_per_category=cfg.max_hacking_pairs_per_category,
            )
        except Exception as exc:
            manifest_rows.append(error_manifest_row(source, str(meta["eval_split"]), "hacking_eval", exc))
            continue
        hacking_records.extend(rows)
        manifest_rows.append(manifest_row(source, str(meta["eval_split"]), "hacking_eval", rows))

    return train_by_source, source_eval_by_source, eval_records, hacking_records, manifest_rows


def init_model_and_tokenizer(cfg: TransformerRMConfig, torch, AutoModelForSequenceClassification, AutoTokenizer, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=1)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if cfg.freeze_backbone:
        freeze_all_but_reward_head(model)
    model.to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return model, tokenizer


def freeze_all_but_reward_head(model) -> None:
    trainable_markers = ("classifier", "score", "reward", "regression", "pre_classifier")
    for name, param in model.named_parameters():
        param.requires_grad = any(marker in name for marker in trainable_markers)


def optimizer_groups(model, weight_decay: float) -> list[dict[str, object]]:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


def encode_texts(tokenizer, texts: list[str], max_length: int, device):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in batch.items()}


def reward_from_batch(model, batch) -> Any:
    return model(**batch).logits.squeeze(-1)


def train_one_model(
    cfg: TransformerRMConfig,
    records: list[PairRecord],
    weight_decay: float,
    seed: int,
    torch,
    F,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    device,
    precision: str,
) -> tuple[Any, Any, list[dict[str, object]]]:
    set_seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model, tokenizer = init_model_and_tokenizer(cfg, torch, AutoModelForSequenceClassification, AutoTokenizer, device)

    optimizer = torch.optim.AdamW(optimizer_groups(model, weight_decay), lr=cfg.lr)
    steps_per_epoch = math.ceil(len(records) / max(cfg.batch_size, 1))
    total_steps = max(math.ceil(steps_per_epoch * cfg.epochs / max(cfg.gradient_accumulation_steps, 1)), 1)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    history: list[dict[str, object]] = []

    rng = random.Random(seed)
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        shuffled = list(records)
        rng.shuffle(shuffled)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        accs = []
        for batch_idx, batch_records in enumerate(batched(shuffled, cfg.batch_size), start=1):
            texts = [r.chosen_text() for r in batch_records] + [r.rejected_text() for r in batch_records]
            enc = encode_texts(tokenizer, texts, cfg.max_length, device)
            with autocast_context(torch, device, precision):
                rewards = reward_from_batch(model, enc)
                chosen_rewards, rejected_rewards = rewards.chunk(2)
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                scaled_loss = loss / cfg.gradient_accumulation_steps
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            margin = (chosen_rewards - rejected_rewards).detach()
            losses.append(float(loss.detach().cpu()))
            accs.append(float((margin > 0).float().mean().cpu()))

            if batch_idx % cfg.gradient_accumulation_steps == 0 or batch_idx == steps_per_epoch:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        history.append(
            {
                "seed": seed,
                "weight_decay": weight_decay,
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(np.mean(losses)),
                "train_accuracy": float(np.mean(accs)),
            }
        )
        print(
            f"seed={seed} wd={weight_decay:g} epoch={epoch + 1}/{cfg.epochs} "
            f"loss={history[-1]['train_loss']:.4f} acc={history[-1]['train_accuracy']:.4f}"
        )
    return model, tokenizer, history


def reward_texts(model, tokenizer, texts: list[str], cfg: TransformerRMConfig, batch_size: int, device, torch, precision: str) -> np.ndarray:
    model.eval()
    rewards = []
    with torch.no_grad():
        for batch_texts in batched(texts, batch_size):
            enc = encode_texts(tokenizer, batch_texts, cfg.max_length, device)
            with autocast_context(torch, device, precision):
                batch_rewards = reward_from_batch(model, enc)
            rewards.extend(batch_rewards.detach().float().cpu().numpy().tolist())
    return np.asarray(rewards, dtype=np.float64)


def evaluate_pairs(
    model,
    tokenizer,
    records: list[PairRecord],
    cfg: TransformerRMConfig,
    batch_size: int,
    device,
    torch,
    precision: str,
    style_scorer: HashingTextFeaturizer,
) -> dict[str, float]:
    chosen_texts = [r.chosen_text() for r in records]
    rejected_texts = [r.rejected_text() for r in records]
    chosen_rewards = reward_texts(model, tokenizer, chosen_texts, cfg, batch_size, device, torch, precision)
    rejected_rewards = reward_texts(model, tokenizer, rejected_texts, cfg, batch_size, device, torch, precision)
    margins = chosen_rewards - rejected_rewards
    chosen_style = np.asarray([style_scorer.style_score(t) for t in chosen_texts], dtype=np.float64)
    rejected_style = np.asarray([style_scorer.style_score(t) for t in rejected_texts], dtype=np.float64)
    return {
        "accuracy": float(np.mean(margins > 0.0)),
        "failure_rate": float(np.mean(margins <= 0.0)),
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
        "chosen_reward_mean": float(np.mean(chosen_rewards)),
        "rejected_reward_mean": float(np.mean(rejected_rewards)),
        "rejected_minus_chosen_style_score": float(np.mean(rejected_style - chosen_style)),
        "reward_style_corr": corr(np.concatenate([chosen_rewards, rejected_rewards]), np.concatenate([chosen_style, rejected_style])),
    }


def evaluate_grouped_pairs(
    model,
    tokenizer,
    records: list[PairRecord],
    cfg: TransformerRMConfig,
    batch_size: int,
    keys: list[str],
    base_labels: dict[str, object],
    device,
    torch,
    precision: str,
    style_scorer: HashingTextFeaturizer,
) -> list[dict[str, object]]:
    rows = []
    for key, group in group_by(records, keys).items():
        if not group:
            continue
        labels = dict(base_labels)
        labels.update(dict(zip(keys, key)))
        labels["n_pairs"] = len(group)
        rows.append(
            {
                **labels,
                **evaluate_pairs(model, tokenizer, group, cfg, batch_size, device, torch, precision, style_scorer),
            }
        )
    return rows


def agop_top_direction(
    model,
    tokenizer,
    records: list[PairRecord],
    cfg: TransformerRMConfig,
    sample_size: int,
    seed: int,
    device,
    torch,
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

    model.eval()
    grads = []
    batch_size = max(1, min(cfg.batch_size, 8))
    for batch_texts in batched(texts, batch_size):
        enc = encode_texts(tokenizer, batch_texts, cfg.max_length, device)
        input_ids = enc.pop("input_ids")
        embeds = model.get_input_embeddings()(input_ids).detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        try:
            out = model(inputs_embeds=embeds, **enc).logits.squeeze(-1)
        except TypeError:
            return np.asarray([float("nan")], dtype=np.float64)
        grad = torch.autograd.grad(out.sum(), embeds, retain_graph=False, create_graph=False)[0]
        mask = enc.get("attention_mask")
        if mask is None:
            pooled = grad.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).to(dtype=grad.dtype)
            pooled = (grad * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        grads.append(pooled.detach().float().cpu().numpy())
    if not grads:
        return np.asarray([float("nan")], dtype=np.float64)
    return top_direction_power(np.concatenate(grads, axis=0))


def top_direction_power(mat: np.ndarray, n_iter: int = 30) -> np.ndarray:
    if mat.ndim != 2 or mat.shape[0] < 2 or not np.all(np.isfinite(mat)):
        return np.asarray([float("nan")], dtype=np.float64)
    centered = mat - mat.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        centered = mat
    rng = np.random.default_rng(123)
    v = rng.normal(size=centered.shape[1])
    v /= max(float(np.linalg.norm(v)), EPS)
    for _ in range(n_iter):
        v = centered.T @ (centered @ v)
        v /= max(float(np.linalg.norm(v)), EPS)
    return v


def safe_abs_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    return abs(cosine(a, b))


def best_of_n_rows(
    model,
    tokenizer,
    records: list[PairRecord],
    cfg: TransformerRMConfig,
    seed: int,
    weight_decay: float,
    device,
    torch,
    precision: str,
    style_scorer: HashingTextFeaturizer,
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
            candidate_texts: list[str] = []
            for rec in trial_records:
                candidate_texts.append(rec.chosen_text())
                for _ in range(n - 1):
                    candidate_texts.append(rng.choice(hacks))
            rewards = reward_texts(model, tokenizer, candidate_texts, cfg, cfg.batch_size * 2, device, torch, precision)
            rewards = rewards.reshape(len(trial_records), n)
            selected_idx = np.argmax(rewards, axis=1)
            selected_texts = [candidate_texts[i * n + int(selected_idx[i])] for i in range(len(trial_records))]
            clean_selected = selected_idx == 0
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
                    "selected_reward_mean": float(np.mean(rewards[np.arange(len(trial_records)), selected_idx])),
                    "selected_style_score_mean": float(np.mean([style_scorer.style_score(t) for t in selected_texts])),
                }
            )
    return rows


def sample_records(source_eval_by_source: dict[str, list[PairRecord]], train_records: list[PairRecord], n: int, seed: int) -> list[PairRecord]:
    rng = random.Random(seed)
    records = flatten(source_eval_by_source.values()) or train_records
    if len(records) <= n:
        return records
    return rng.sample(records, n)


def run_workflow(cfg: TransformerRMConfig) -> None:
    torch, F, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup = require_torch_transformers()
    device = resolve_device(torch, cfg.device)
    precision = resolve_precision(torch, device, cfg.precision)
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    device_info = {"device": str(device), "precision": precision}
    if device.type == "cuda":
        device_info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }
        )
    (outdir / "device_info.json").write_text(json.dumps(device_info, indent=2), encoding="utf-8")
    print(f"Using device={device_info}")

    train_by_source, source_eval_by_source, eval_records, hacking_records, manifest_rows = load_all_records(cfg)
    train_records = flatten(train_by_source.values())
    if not train_records:
        raise SystemExit("No training records loaded. Check dataset names/splits and network access.")
    if not hacking_records:
        raise SystemExit("No reward-hacking evaluation records loaded. Check rh-bench adapters.")

    style_scorer = HashingTextFeaturizer(hash_dim=8, max_tokens=1, max_chars=1)
    write_csv(outdir / "dataset_manifest.csv", manifest_rows)

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
            model, tokenizer, history = train_one_model(
                cfg,
                train_sample,
                wd,
                seed,
                torch,
                F,
                AutoModelForSequenceClassification,
                AutoTokenizer,
                get_linear_schedule_with_warmup,
                device,
                precision,
            )
            train_history_rows.extend(history)

            for source, records in source_eval_by_source.items():
                source_eval_rows.extend(
                    evaluate_grouped_pairs(
                        model,
                        tokenizer,
                        records,
                        cfg,
                        cfg.batch_size * 2,
                        keys=["dataset", "category"],
                        base_labels={"seed": seed, "weight_decay": wd, "eval_kind": "source", "source": source},
                        device=device,
                        torch=torch,
                        precision=precision,
                        style_scorer=style_scorer,
                    )
                )

            preference_eval_rows.extend(
                evaluate_grouped_pairs(
                    model,
                    tokenizer,
                    eval_records,
                    cfg,
                    cfg.batch_size * 2,
                    keys=["dataset", "category"],
                    base_labels={"seed": seed, "weight_decay": wd, "eval_kind": "ood_preference"},
                    device=device,
                    torch=torch,
                    precision=precision,
                    style_scorer=style_scorer,
                )
            )

            hacking_eval_rows.extend(
                evaluate_grouped_pairs(
                    model,
                    tokenizer,
                    hacking_records,
                    cfg,
                    cfg.batch_size * 2,
                    keys=["dataset", "category"],
                    base_labels={"seed": seed, "weight_decay": wd, "eval_kind": "reward_hacking", "positive_label": "clean"},
                    device=device,
                    torch=torch,
                    precision=precision,
                    style_scorer=style_scorer,
                )
            )

            source_agop_records = sample_records(source_eval_by_source, train_sample, cfg.agop_sample_size, seed=30_000 + seed)
            source_top = agop_top_direction(model, tokenizer, source_agop_records, cfg, cfg.agop_sample_size, 40_000 + seed, device, torch)
            for (dataset, category), group in group_by(hacking_records, ["dataset", "category"]).items():
                target_top = agop_top_direction(model, tokenizer, group, cfg, cfg.agop_sample_size, 50_000 + seed, device, torch)
                agop_rows.append(
                    {
                        "seed": seed,
                        "weight_decay": wd,
                        "dataset": dataset,
                        "category": category,
                        "source_target_agop_abs_cosine": safe_abs_cosine(source_top, target_top),
                        "agop_embedding_dim": int(source_top.size),
                    }
                )

            bon_rows.extend(
                best_of_n_rows(
                    model,
                    tokenizer,
                    hacking_records,
                    cfg,
                    seed=60_000 + seed,
                    weight_decay=wd,
                    device=device,
                    torch=torch,
                    precision=precision,
                    style_scorer=style_scorer,
                )
            )

            write_csv(outdir / "train_history.csv", train_history_rows)
            write_csv(outdir / "source_eval.csv", source_eval_rows)
            write_csv(outdir / "ood_preference_eval.csv", preference_eval_rows)
            write_csv(outdir / "hacking_eval.csv", hacking_eval_rows)
            write_csv(outdir / "agop_diagnostics.csv", agop_rows)
            write_csv(outdir / "best_of_n.csv", bon_rows)
            write_csv(outdir / "summary_by_weight_decay.csv", summary_by_weight_decay(hacking_eval_rows, bon_rows, cfg))

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_report(outdir / "report.md", cfg, manifest_rows, source_eval_rows, preference_eval_rows, hacking_eval_rows, agop_rows, bon_rows, device_info)
    print(f"Wrote transformer RM workflow results to {outdir.resolve()}")


def summary_by_weight_decay(
    hacking_rows: list[dict[str, object]],
    bon_rows: list[dict[str, object]],
    cfg: TransformerRMConfig,
) -> list[dict[str, object]]:
    hack_summary = aggregate(hacking_rows, ["weight_decay"])
    bon_max = [r for r in bon_rows if int(r["N"]) == max(cfg.n_values)]
    bon_summary = aggregate(bon_max, ["weight_decay"])
    return merge_on_keys(hack_summary, bon_summary, ["weight_decay"])


def write_report(
    path: Path,
    cfg: TransformerRMConfig,
    manifest_rows: list[dict[str, object]],
    source_eval_rows: list[dict[str, object]],
    preference_eval_rows: list[dict[str, object]],
    hacking_eval_rows: list[dict[str, object]],
    agop_rows: list[dict[str, object]],
    bon_rows: list[dict[str, object]],
    device_info: dict[str, object],
) -> None:
    baseline_wd = min(cfg.weight_decays)
    max_n = max(cfg.n_values)
    h1 = aggregate([r for r in hacking_eval_rows if float(r["weight_decay"]) == baseline_wd], ["dataset", "category"])
    source = aggregate([r for r in source_eval_rows if float(r["weight_decay"]) == baseline_wd], ["dataset", "category"])
    pref = aggregate([r for r in preference_eval_rows if float(r["weight_decay"]) == baseline_wd], ["dataset", "category"])
    agop = aggregate([r for r in agop_rows if float(r["weight_decay"]) == baseline_wd], ["dataset", "category"])
    h3 = aggregate(hacking_eval_rows, ["dataset", "category", "weight_decay"])
    bon_max = aggregate([r for r in bon_rows if int(r["N"]) == max_n], ["dataset", "category", "weight_decay"])
    h3_merged = merge_on_keys(h3, bon_max, ["dataset", "category", "weight_decay"])
    best_h3 = best_by_group(h3_merged, ["dataset", "category"], metric="hacking_selected_rate_mean", maximize=False)
    wd_summary = summary_by_weight_decay(hacking_eval_rows, bon_rows, cfg)

    lines = [
        "# Transformer Reward Model Workflow Report",
        "",
        "## Architecture",
        "",
        f"- Model: `{cfg.model_name}` with `AutoModelForSequenceClassification(num_labels=1)`.",
        "- Reward head: scalar score for each prompt-response text.",
        "- Loss: Bradley-Terry pairwise preference loss, `-logsigmoid(r_chosen - r_rejected)`.",
        f"- Device: `{device_info}`.",
        "",
        "## Dataset Manifest",
        "",
        markdown_table(manifest_rows, ["source_key", "hf_name", "split", "role", "n_pairs", "n_categories", "license"], max_rows=24),
        "",
        "## H1: reward hacking on open benchmarks",
        "",
        markdown_table(
            h1,
            ["dataset", "category", "n_pairs_mean", "accuracy_mean", "failure_rate_mean", "margin_mean_mean", "reward_style_corr_mean"],
            max_rows=30,
        ),
        "",
        "## H2: AGOP transfer in token-embedding space",
        "",
        markdown_table(
            agop,
            ["dataset", "category", "source_target_agop_abs_cosine_mean", "agop_embedding_dim_mean"],
            max_rows=30,
        ),
        "",
        "## H3: weight decay sweep",
        "",
        f"Rows are selected by minimum hacking-selected rate under best-of-N at N={max_n}.",
        "",
        markdown_table(
            best_h3,
            ["dataset", "category", "weight_decay", "accuracy_mean", "failure_rate_mean", "hacking_selected_rate_mean", "clean_selected_rate_mean"],
            max_rows=30,
        ),
        "",
        f"## Aggregate Weight-Decay Trend at N={max_n}",
        "",
        markdown_table(
            wd_summary,
            ["weight_decay", "accuracy_mean", "failure_rate_mean", "hacking_selected_rate_mean", "clean_selected_rate_mean"],
            max_rows=20,
        ),
        "",
        "## Source and OOD preference checks",
        "",
        markdown_table(source, ["dataset", "category", "accuracy_mean", "margin_mean_mean"], max_rows=20),
        "",
        markdown_table(pref, ["dataset", "category", "accuracy_mean", "margin_mean_mean"], max_rows=20),
        "",
        "## Files",
        "",
        "- `device_info.json`: CUDA/GPU and precision metadata.",
        "- `dataset_manifest.csv`: loaded open datasets and category counts.",
        "- `train_history.csv`: pairwise RM training loss and accuracy.",
        "- `source_eval.csv`: held-out source preference accuracy.",
        "- `ood_preference_eval.csv`: RewardBench-style OOD preference accuracy.",
        "- `hacking_eval.csv`: clean-vs-hacking pair accuracy by category.",
        "- `agop_diagnostics.csv`: source-target AGOP cosine in embedding-gradient space.",
        "- `best_of_n.csv`: adversarial clean-vs-many-hacking selection curves.",
        "- `summary_by_weight_decay.csv`: aggregate mitigation trend.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate a GPU transformer reward model on open reward-hacking data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Lightweight:
              python src/transformer_rm_workflow.py --quick --device cuda

            Full:
              python src/transformer_rm_workflow.py \\
                --output-dir results/transformer_rm_full \\
                --model-name distilroberta-base \\
                --seeds 0 1 2 \\
                --weight-decays 0 1e-5 1e-4 1e-3 1e-2 \\
                --device cuda
            """
        ),
    )
    p.add_argument("--output-dir", default="results/transformer_rm_full")
    p.add_argument("--quick", action="store_true", help="Single-seed, small-data GPU smoke experiment.")
    p.add_argument("--model-name", default="distilroberta-base")
    p.add_argument("--train-sources", nargs="+", default=["ultrafeedback", "hh_rlhf"])
    p.add_argument("--eval-sources", nargs="+", default=["rewardbench"])
    p.add_argument("--hacking-sources", nargs="+", default=["rh_bench_open", "rh_bench_multi"])
    p.add_argument("--max-train-pairs-per-source", type=int, default=12_000)
    p.add_argument("--max-eval-pairs-per-dataset", type=int, default=4_000)
    p.add_argument("--max-hacking-pairs-per-category", type=int, default=1_200)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    p.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--bon-trials-per-category", type=int, default=400)
    p.add_argument("--agop-sample-size", type=int, default=128)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--freeze-backbone", action="store_true", help="Train only the scalar reward head.")
    p.add_argument("--no-streaming", action="store_true", help="Download full HF datasets instead of streaming.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.output_dir = "results/transformer_rm_smoke" if args.output_dir == "results/transformer_rm_full" else args.output_dir
        args.max_train_pairs_per_source = 500
        args.max_eval_pairs_per_dataset = 300
        args.max_hacking_pairs_per_category = 100
        args.epochs = 1
        args.batch_size = 8
        args.gradient_accumulation_steps = 2
        args.max_length = 384
        args.seeds = [0]
        args.weight_decays = [0.0, 1e-3, 1e-2]
        args.n_values = [1, 2, 4, 8]
        args.bon_trials_per_category = 60
        args.agop_sample_size = 32

    cfg = TransformerRMConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_sources=tuple(args.train_sources),
        eval_sources=tuple(args.eval_sources),
        hacking_sources=tuple(args.hacking_sources),
        max_train_pairs_per_source=args.max_train_pairs_per_source,
        max_eval_pairs_per_dataset=args.max_eval_pairs_per_dataset,
        max_hacking_pairs_per_category=args.max_hacking_pairs_per_category,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        seeds=tuple(args.seeds),
        weight_decays=tuple(args.weight_decays),
        n_values=tuple(args.n_values),
        bon_trials_per_category=args.bon_trials_per_category,
        agop_sample_size=args.agop_sample_size,
        device=args.device,
        precision=args.precision,
        freeze_backbone=args.freeze_backbone,
        streaming=not args.no_streaming,
        quick=args.quick,
    )
    run_workflow(cfg)


if __name__ == "__main__":
    main()
