#!/usr/bin/env python3
"""Small group-meeting workflow for superposition-driven reward hacking.

This script is intentionally lightweight. It uses only NumPy/Pandas so the
workflow can run on a laptop without a GPU or Hugging Face model downloads.

It runs two linked experiments:

H1/H2 diagnostics:
    Across several proxy-reward domains and reward-model variants, train a
    small bottleneck reward model on a source distribution where true utility
    and a proxy feature are correlated. Then test whether AGOP directions stay
    aligned across target/conflict domains where the proxy becomes harmful.

H3 mitigation:
    For the same domains, sweep decoupled weight decay while holding model size
    fixed. Report whether weight decay reduces spurious sensitivity, AGOP
    transfer, and best-of-N reward hacking pressure.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class DomainSpec:
    name: str
    reward: str
    source_domain: str
    target_domain: str
    train_rho: float
    target_rho: float
    conflict_strength: float
    true_noise: float
    spur_noise: float
    hack_style_scale: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    bottleneck_dim: int
    hidden_scale: float
    train_pairs: int
    epochs: int
    lr: float


@dataclass
class WorkflowConfig:
    output_dir: str = "results/group_meeting_smoke"
    train_pairs: int = 1200
    test_pairs: int = 800
    epochs: int = 70
    batch_size: int = 256
    seeds: tuple[int, ...] = (0, 1)
    weight_decays: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 5e-1, 1.0)
    n_values: tuple[int, ...] = (1, 4, 16, 64)
    bon_prompts: int = 400
    true_dim: int = 6
    spur_dim: int = 14
    noise_dim: int = 12
    quick: bool = False

    @property
    def x_dim(self) -> int:
        return self.true_dim + self.spur_dim + self.noise_dim


def domain_suite() -> list[DomainSpec]:
    return [
        DomainSpec(
            name="verbosity",
            reward="helpfulness",
            source_domain="long_explanations_help",
            target_domain="concise_answers_required",
            train_rho=0.92,
            target_rho=-0.85,
            conflict_strength=2.2,
            true_noise=0.55,
            spur_noise=0.18,
            hack_style_scale=2.5,
        ),
        DomainSpec(
            name="formatting",
            reward="readability",
            source_domain="structured_lists_help",
            target_domain="formatting_hides_wrong_answer",
            train_rho=0.88,
            target_rho=-0.80,
            conflict_strength=2.0,
            true_noise=0.50,
            spur_noise=0.22,
            hack_style_scale=2.2,
        ),
        DomainSpec(
            name="sycophancy",
            reward="supportiveness",
            source_domain="emotional_support",
            target_domain="false_user_belief",
            train_rho=0.93,
            target_rho=-0.90,
            conflict_strength=2.4,
            true_noise=0.60,
            spur_noise=0.18,
            hack_style_scale=2.7,
        ),
        DomainSpec(
            name="confidence",
            reward="decisiveness",
            source_domain="urgent_safety_advice",
            target_domain="unknown_fact_hallucination",
            train_rho=0.95,
            target_rho=-0.92,
            conflict_strength=2.5,
            true_noise=0.70,
            spur_noise=0.16,
            hack_style_scale=3.0,
        ),
        DomainSpec(
            name="fabricated_reasoning",
            reward="reasoning_quality",
            source_domain="correct_step_by_step",
            target_domain="plausible_but_false_rationale",
            train_rho=0.90,
            target_rho=-0.88,
            conflict_strength=2.3,
            true_noise=0.65,
            spur_noise=0.20,
            hack_style_scale=2.8,
        ),
        DomainSpec(
            name="code_test_hacking",
            reward="test_passing",
            source_domain="general_code_quality",
            target_domain="visible_test_overfitting",
            train_rho=0.86,
            target_rho=-0.78,
            conflict_strength=2.0,
            true_noise=0.55,
            spur_noise=0.25,
            hack_style_scale=2.1,
        ),
    ]


def mitigation_domain_suite() -> list[DomainSpec]:
    """Cleaner H3 suite where true signal exists but proxy shortcuts tempt the RM."""
    return [
        DomainSpec(
            name="verbosity",
            reward="helpfulness",
            source_domain="long_explanations_help",
            target_domain="concise_answers_required",
            train_rho=0.92,
            target_rho=-0.85,
            conflict_strength=2.2,
            true_noise=0.30,
            spur_noise=0.18,
            hack_style_scale=2.5,
        ),
        DomainSpec(
            name="formatting",
            reward="readability",
            source_domain="structured_lists_help",
            target_domain="formatting_hides_wrong_answer",
            train_rho=0.88,
            target_rho=-0.80,
            conflict_strength=2.0,
            true_noise=0.28,
            spur_noise=0.22,
            hack_style_scale=2.2,
        ),
        DomainSpec(
            name="confidence",
            reward="decisiveness",
            source_domain="urgent_safety_advice",
            target_domain="unknown_fact_hallucination",
            train_rho=0.95,
            target_rho=-0.92,
            conflict_strength=2.5,
            true_noise=0.35,
            spur_noise=0.16,
            hack_style_scale=3.0,
        ),
        DomainSpec(
            name="fabricated_reasoning",
            reward="reasoning_quality",
            source_domain="correct_step_by_step",
            target_domain="plausible_but_false_rationale",
            train_rho=0.90,
            target_rho=-0.88,
            conflict_strength=2.3,
            true_noise=0.33,
            spur_noise=0.20,
            hack_style_scale=2.8,
        ),
    ]


def h12_model_suite(cfg: WorkflowConfig) -> list[ModelSpec]:
    return [
        ModelSpec("compact_rm", bottleneck_dim=3, hidden_scale=0.80, train_pairs=cfg.train_pairs, epochs=cfg.epochs, lr=2e-2),
        ModelSpec("balanced_rm", bottleneck_dim=5, hidden_scale=1.00, train_pairs=cfg.train_pairs, epochs=cfg.epochs, lr=2e-2),
        ModelSpec("less_compressed_rm", bottleneck_dim=8, hidden_scale=1.10, train_pairs=cfg.train_pairs, epochs=cfg.epochs, lr=2e-2),
    ]


class SyntheticDomainWorld:
    def __init__(self, cfg: WorkflowConfig, spec: DomainSpec, seed: int) -> None:
        self.cfg = cfg
        self.spec = spec
        rng = np.random.default_rng(seed + 17)
        self.true_w = unit(rng.normal(size=cfg.true_dim))
        spur_raw = rng.normal(size=cfg.spur_dim)
        # Spread the proxy across many coordinates so using it requires a broad
        # representation. This makes weight decay a meaningful knob.
        self.spur_w = unit(spur_raw + 0.35 * np.sign(spur_raw))

    def full_true_dir(self) -> np.ndarray:
        return np.concatenate([self.true_w, np.zeros(self.cfg.spur_dim + self.cfg.noise_dim)])

    def full_spur_dir(self) -> np.ndarray:
        return np.concatenate([np.zeros(self.cfg.true_dim), self.spur_w, np.zeros(self.cfg.noise_dim)])

    def candidates(
        self,
        n: int,
        rho: float,
        rng: np.random.Generator,
        conflict: bool = False,
        hack_distribution: bool = False,
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg
        spec = self.spec
        z_true_clean = rng.normal(size=(n, cfg.true_dim))
        true_u = z_true_clean @ self.true_w + 0.08 * rng.normal(size=n)
        z_true = z_true_clean + spec.true_noise * rng.normal(size=(n, cfg.true_dim))

        eps = rng.normal(size=(n, cfg.spur_dim))
        rho_abs = min(abs(rho), 0.995)
        spur = rho * true_u[:, None] * self.spur_w[None, :] + math.sqrt(1.0 - rho_abs**2) * eps

        if conflict:
            spur = -spec.conflict_strength * true_u[:, None] * self.spur_w[None, :] + 0.25 * eps
        if hack_distribution:
            style = rng.exponential(scale=spec.hack_style_scale, size=n)
            spur = (
                style[:, None] * self.spur_w[None, :]
                - 0.40 * true_u[:, None] * self.spur_w[None, :]
                + 0.30 * eps
            )

        spur = spur + spec.spur_noise * rng.normal(size=(n, cfg.spur_dim))
        nuisance = rng.normal(size=(n, cfg.noise_dim))
        x = np.concatenate([z_true, spur, nuisance], axis=1)
        spur_u = spur @ self.spur_w
        return {"x": x.astype(np.float64), "true_u": true_u.astype(np.float64), "spur_u": spur_u.astype(np.float64)}

    def pairs(self, n_pairs: int, rho: float, seed: int, conflict: bool = False) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        a = self.candidates(n_pairs, rho, rng, conflict=conflict)
        b = self.candidates(n_pairs, rho, rng, conflict=conflict)
        choose_a = a["true_u"] >= b["true_u"]
        return {
            "x_chosen": np.where(choose_a[:, None], a["x"], b["x"]),
            "x_rejected": np.where(choose_a[:, None], b["x"], a["x"]),
            "chosen_true": np.where(choose_a, a["true_u"], b["true_u"]),
            "rejected_true": np.where(choose_a, b["true_u"], a["true_u"]),
            "chosen_spur": np.where(choose_a, a["spur_u"], b["spur_u"]),
            "rejected_spur": np.where(choose_a, b["spur_u"], a["spur_u"]),
        }

    def sensitivity_batch(self, n: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        cfg = self.cfg
        spec = self.spec
        z_true = rng.normal(size=(n, cfg.true_dim))
        nuisance = rng.normal(size=(n, cfg.noise_dim))
        low_true = -spec.conflict_strength * self.true_w[None, :].repeat(n, axis=0)
        high_true = spec.conflict_strength * self.true_w[None, :].repeat(n, axis=0)
        low_spur = -spec.conflict_strength * self.spur_w[None, :].repeat(n, axis=0)
        high_spur = spec.conflict_strength * self.spur_w[None, :].repeat(n, axis=0)
        neutral_spur = np.zeros((n, cfg.spur_dim))
        return {
            "x_low_true": np.concatenate([low_true, neutral_spur, nuisance], axis=1),
            "x_high_true": np.concatenate([high_true, neutral_spur, nuisance], axis=1),
            "x_low_spur": np.concatenate([z_true, low_spur, nuisance], axis=1),
            "x_high_spur": np.concatenate([z_true, high_spur, nuisance], axis=1),
        }

    def best_of_n_pool(self, n_prompts: int, n_candidates: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        flat = self.candidates(n_prompts * n_candidates, self.spec.target_rho, rng, hack_distribution=True)
        return {
            "x": flat["x"].reshape(n_prompts, n_candidates, -1),
            "true_u": flat["true_u"].reshape(n_prompts, n_candidates),
            "spur_u": flat["spur_u"].reshape(n_prompts, n_candidates),
        }


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
        data: dict[str, np.ndarray],
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        n = data["x_chosen"].shape[0]
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                grads = self._batch_grads(data["x_chosen"][idx], data["x_rejected"][idx])
                self._adamw(grads, lr=lr, weight_decay=weight_decay)

    def _batch_grads(self, x_c: np.ndarray, x_r: np.ndarray) -> list[np.ndarray]:
        z_c = self.encode(x_c)
        z_r = self.encode(x_r)
        margin = z_c @ self.v - z_r @ self.v
        d_margin = sigmoid(margin) - 1.0
        scale = 1.0 / max(x_c.shape[0], 1)
        d_c = d_margin * scale
        d_r = -d_margin * scale

        grad_v = z_c.T @ d_c + z_r.T @ d_r
        ga_c = d_c[:, None] * self.v[None, :] * (1.0 - z_c**2)
        ga_r = d_r[:, None] * self.v[None, :] * (1.0 - z_r**2)
        grad_w = x_c.T @ ga_c + x_r.T @ ga_r
        grad_b = ga_c.sum(axis=0) + ga_r.sum(axis=0)
        return [grad_w, grad_b, grad_v]

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


def run_workflow(cfg: WorkflowConfig) -> None:
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    h12_rows: list[dict[str, float | str | bool]] = []
    h3_rows: list[dict[str, float | str | bool]] = []
    bon_rows: list[dict[str, float | str]] = []

    h12_domains = domain_suite()
    h3_domains = mitigation_domain_suite()
    h12_models = h12_model_suite(cfg)

    for seed in cfg.seeds:
        for domain_idx, spec in enumerate(h12_domains):
            world = SyntheticDomainWorld(cfg, spec, seed=1000 * seed + domain_idx)
            source_train = world.pairs(cfg.train_pairs, spec.train_rho, seed=10_000 + seed + domain_idx)
            source_eval = world.pairs(cfg.test_pairs, spec.train_rho, seed=20_000 + seed + domain_idx)
            target_eval = world.pairs(cfg.test_pairs, spec.target_rho, seed=30_000 + seed + domain_idx)
            conflict_eval = world.pairs(cfg.test_pairs, spec.target_rho, seed=40_000 + seed + domain_idx, conflict=True)

            for model_idx, model_spec in enumerate(h12_models):
                model = NumpyRewardModel(
                    cfg.x_dim,
                    bottleneck_dim=model_spec.bottleneck_dim,
                    hidden_scale=model_spec.hidden_scale,
                    seed=50_000 + seed * 101 + domain_idx * 13 + model_idx,
                )
                model.fit(
                    source_train,
                    epochs=model_spec.epochs,
                    batch_size=cfg.batch_size,
                    lr=model_spec.lr,
                    weight_decay=0.0,
                    seed=60_000 + seed,
                )
                h12_rows.append(
                    {
                        **base_labels(seed, spec),
                        "reward_model": model_spec.name,
                        "bottleneck_dim": model_spec.bottleneck_dim,
                        **diagnostics(model, world, source_eval, target_eval, conflict_eval, cfg, seed),
                    }
                )

        # H3 uses the compact model and a cleaner suite to keep the mitigation
        # claim separate from the stronger H1/H2 domain-shift diagnostic.
        sweep_spec = h12_models[0]
        for domain_idx, spec in enumerate(h3_domains):
            world = SyntheticDomainWorld(cfg, spec, seed=2000 * seed + domain_idx)
            source_train = world.pairs(cfg.train_pairs, spec.train_rho, seed=110_000 + seed + domain_idx)
            source_eval = world.pairs(cfg.test_pairs, spec.train_rho, seed=120_000 + seed + domain_idx)
            target_eval = world.pairs(cfg.test_pairs, spec.target_rho, seed=130_000 + seed + domain_idx)
            conflict_eval = world.pairs(cfg.test_pairs, spec.target_rho, seed=140_000 + seed + domain_idx, conflict=True)
            for wd in cfg.weight_decays:
                model = NumpyRewardModel(
                    cfg.x_dim,
                    bottleneck_dim=sweep_spec.bottleneck_dim,
                    hidden_scale=sweep_spec.hidden_scale,
                    seed=150_000 + seed * 101 + domain_idx,
                )
                model.fit(
                    source_train,
                    epochs=sweep_spec.epochs,
                    batch_size=cfg.batch_size,
                    lr=sweep_spec.lr,
                    weight_decay=wd,
                    seed=160_000 + seed,
                )
                row = {
                    **base_labels(seed, spec),
                    "reward_model": sweep_spec.name,
                    "weight_decay": wd,
                    "bottleneck_dim": sweep_spec.bottleneck_dim,
                    **diagnostics(model, world, source_eval, target_eval, conflict_eval, cfg, seed),
                }
                h3_rows.append(row)
                bon_rows.extend(best_of_n_rows(model, world, cfg, seed=170_000 + seed, weight_decay=wd))

    write_csv(outdir / "h12_diagnostics.csv", h12_rows)
    write_csv(outdir / "h3_weight_decay.csv", h3_rows)
    write_csv(outdir / "h3_best_of_n.csv", bon_rows)
    write_markdown_report(outdir / "report.md", h12_rows, h3_rows, bon_rows, cfg)
    maybe_write_plots(outdir, h3_rows, bon_rows)


def base_labels(seed: int, spec: DomainSpec) -> dict[str, str | int | float]:
    return {
        "seed": seed,
        "phenomenon": spec.name,
        "reward": spec.reward,
        "source_domain": spec.source_domain,
        "target_domain": spec.target_domain,
        "train_rho": spec.train_rho,
        "target_rho": spec.target_rho,
    }


def diagnostics(
    model: NumpyRewardModel,
    world: SyntheticDomainWorld,
    source_eval: dict[str, np.ndarray],
    target_eval: dict[str, np.ndarray],
    conflict_eval: dict[str, np.ndarray],
    cfg: WorkflowConfig,
    seed: int,
) -> dict[str, float | bool]:
    rng = np.random.default_rng(110_000 + seed)
    source_cand = world.candidates(cfg.test_pairs, world.spec.train_rho, rng)
    target_cand = world.candidates(cfg.test_pairs, world.spec.target_rho, rng, conflict=True)
    sens = world.sensitivity_batch(cfg.test_pairs, seed=120_000 + seed)

    source_grad = model.input_grad(source_cand["x"])
    target_grad = model.input_grad(target_cand["x"])
    source_top = top_agop_direction(source_grad)
    target_top = top_agop_direction(target_grad)

    source_reward = model.reward(source_cand["x"])
    target_reward = model.reward(target_cand["x"])
    source_true_corr = corr(source_reward, source_cand["true_u"])
    target_true_corr = corr(target_reward, target_cand["true_u"])
    target_spur_corr = corr(target_reward, target_cand["spur_u"])
    agop_abs_cos = abs(cosine(source_top, target_top))
    transfer_score = (
        max(source_true_corr, 0.0)
        * max(-target_true_corr, 0.0)
        * max(target_spur_corr, 0.0)
        * agop_abs_cos
    )

    low_spur = model.reward(sens["x_low_spur"])
    high_spur = model.reward(sens["x_high_spur"])
    low_true = model.reward(sens["x_low_true"])
    high_true = model.reward(sens["x_high_true"])
    spur_sens = float(np.mean(high_spur - low_spur))
    true_sens = float(np.mean(high_true - low_true))
    ratio = abs(spur_sens) / max(abs(true_sens), EPS)

    z = model.encode(target_cand["x"])
    true_probe = ridge_probe(z, target_cand["true_u"])
    spur_probe = ridge_probe(z, target_cand["spur_u"])

    return {
        "source_accuracy": pair_accuracy(model, source_eval),
        "target_accuracy": pair_accuracy(model, target_eval),
        "conflict_accuracy": pair_accuracy(model, conflict_eval),
        "source_reward_true_corr": source_true_corr,
        "target_reward_true_corr": target_true_corr,
        "target_reward_spur_corr": target_spur_corr,
        "agop_source_target_abs_cosine": agop_abs_cos,
        "agop_top_true_alignment": abs(cosine(target_top, world.full_true_dir())),
        "agop_top_spur_alignment": abs(cosine(target_top, world.full_spur_dir())),
        "transfer_spurious_score": transfer_score,
        "label_flip_signal": bool(source_true_corr > 0 and target_true_corr < 0),
        "shared_direction_signal": bool(agop_abs_cos >= 0.65),
        "spurious_sensitivity": spur_sens,
        "true_sensitivity": true_sens,
        "abs_spurious_over_true_sensitivity": ratio,
        "probe_abs_cos_true_spur": abs(cosine(true_probe, spur_probe)),
        "probe_r2_true": probe_r2(z, target_cand["true_u"], true_probe),
        "probe_r2_spur": probe_r2(z, target_cand["spur_u"], spur_probe),
    }


def best_of_n_rows(
    model: NumpyRewardModel,
    world: SyntheticDomainWorld,
    cfg: WorkflowConfig,
    seed: int,
    weight_decay: float,
) -> list[dict[str, float | str]]:
    n_max = max(cfg.n_values)
    pool = world.best_of_n_pool(cfg.bon_prompts, n_max, seed)
    flat_x = pool["x"].reshape(cfg.bon_prompts * n_max, -1)
    rewards = model.reward(flat_x).reshape(cfg.bon_prompts, n_max)
    rows = []
    row_idx = np.arange(cfg.bon_prompts)
    for n in cfg.n_values:
        selected_idx = np.argmax(rewards[:, :n], axis=1)
        oracle_idx = np.argmax(pool["true_u"][:, :n], axis=1)
        selected_true = pool["true_u"][row_idx, selected_idx]
        selected_spur = pool["spur_u"][row_idx, selected_idx]
        selected_reward = rewards[row_idx, selected_idx]
        rows.append(
            {
                "seed": seed,
                "phenomenon": world.spec.name,
                "reward": world.spec.reward,
                "weight_decay": weight_decay,
                "N": n,
                "selected_proxy_reward_mean": float(np.mean(selected_reward)),
                "selected_true_utility_mean": float(np.mean(selected_true)),
                "selected_spurious_utility_mean": float(np.mean(selected_spur)),
                "oracle_best_true_utility_mean": float(np.mean(pool["true_u"][row_idx, oracle_idx])),
                "proxy_minus_true_gap": float(np.mean(selected_reward - selected_true)),
            }
        )
    return rows


def pair_accuracy(model: NumpyRewardModel, data: dict[str, np.ndarray]) -> float:
    margin = model.reward(data["x_chosen"]) - model.reward(data["x_rejected"])
    return float(np.mean(margin > 0.0))


def top_agop_direction(grads: np.ndarray) -> np.ndarray:
    centered = grads - grads.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        centered = grads
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def ridge_probe(z: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    zc = z - z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    xtx = zc.T @ zc
    return np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), zc.T @ yc)


def probe_r2(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    zc = z - z.mean(axis=0, keepdims=True)
    pred = zc @ w + y.mean()
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = max(np.sum((y - y.mean()) ** 2), EPS)
    return float(1.0 - ss_res / ss_tot)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def unit(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), EPS)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    if denom < EPS:
        return float("nan")
    return float(np.sum(a * b) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS)
    return float(np.dot(a, b) / denom)


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


def write_markdown_report(
    path: Path,
    h12_rows: list[dict[str, object]],
    h3_rows: list[dict[str, object]],
    bon_rows: list[dict[str, object]],
    cfg: WorkflowConfig,
) -> None:
    h12 = aggregate(h12_rows, ["phenomenon", "reward"])
    h3 = aggregate(h3_rows, ["phenomenon", "reward", "weight_decay"])
    bon64 = [r for r in bon_rows if int(r["N"]) == max(cfg.n_values)]
    bon = aggregate(bon64, ["phenomenon", "reward", "weight_decay"])
    h3_bon = merge_on_keys(h3, bon, ["phenomenon", "reward", "weight_decay"])
    best_h3 = best_by_domain(h3_bon, metric="proxy_minus_true_gap_mean", maximize=False)

    lines = [
        "# Group Meeting Workflow Report",
        "",
        "## H1/H2: multi-domain reward overgeneralization",
        "",
        "A positive row has high source accuracy, low target/conflict accuracy, high AGOP source-target cosine, and positive transfer spurious score.",
        "",
        markdown_table(
            h12,
            [
                "phenomenon",
                "reward",
                "source_accuracy_mean",
                "target_accuracy_mean",
                "conflict_accuracy_mean",
                "agop_source_target_abs_cosine_mean",
                "target_reward_spur_corr_mean",
                "transfer_spurious_score_mean",
            ],
            max_rows=24,
        ),
        "",
        "## H3: best weight decay per phenomenon",
        "",
        f"Rows are selected by minimum proxy-minus-true hacking gap under best-of-N at N={max(cfg.n_values)}. This is the group-meeting mitigation view; the CSV keeps all weight-decay points.",
        "",
        markdown_table(
            best_h3,
            [
                "phenomenon",
                "reward",
                "weight_decay",
                "target_accuracy_mean",
                "conflict_accuracy_mean",
                "abs_spurious_over_true_sensitivity_mean",
                "transfer_spurious_score_mean",
                "selected_true_utility_mean_mean",
                "proxy_minus_true_gap_mean",
            ],
            max_rows=24,
        ),
        "",
        f"## Best-of-N at N={max(cfg.n_values)}",
        "",
        markdown_table(
            bon,
            [
                "phenomenon",
                "reward",
                "weight_decay",
                "selected_proxy_reward_mean_mean",
                "selected_true_utility_mean_mean",
                "selected_spurious_utility_mean_mean",
                "proxy_minus_true_gap_mean",
            ],
            max_rows=36,
        ),
        "",
        "## Files",
        "",
        "- `h12_diagnostics.csv`: one row per seed/domain/reward-model variant.",
        "- `h3_weight_decay.csv`: one row per seed/domain/weight decay.",
        "- `h3_best_of_n.csv`: best-of-N hacking curves.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


def best_by_domain(
    rows: list[dict[str, object]],
    metric: str,
    maximize: bool = False,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, object], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["phenomenon"], row["reward"]), []).append(row)
    best = []
    for _, group_rows in groups.items():
        finite = [r for r in group_rows if metric in r and np.isfinite(float(r[metric]))]
        if not finite:
            continue
        key_fn = lambda r: float(r[metric])
        best.append(max(finite, key=key_fn) if maximize else min(finite, key=key_fn))
    return sorted(best, key=lambda r: str(r["phenomenon"]))


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


def maybe_write_plots(outdir: Path, h3_rows: list[dict[str, object]], bon_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = outdir / "plots"
    plot_dir.mkdir(exist_ok=True)
    h3 = aggregate(h3_rows, ["phenomenon", "weight_decay"])
    for metric in ["transfer_spurious_score_mean", "abs_spurious_over_true_sensitivity_mean", "target_accuracy_mean"]:
        plt.figure(figsize=(8, 5))
        for phenomenon in sorted({str(r["phenomenon"]) for r in h3}):
            sub = [r for r in h3 if r["phenomenon"] == phenomenon]
            xs = [float(r["weight_decay"]) for r in sub]
            ys = [float(r.get(metric, np.nan)) for r in sub]
            plt.plot(xs, ys, marker="o", label=phenomenon)
        plt.xscale("symlog", linthresh=1e-5)
        plt.xlabel("weight_decay")
        plt.ylabel(metric)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small workflow for group-meeting reward hacking experiments.")
    p.add_argument("--output-dir", default="results/group_meeting_smoke")
    p.add_argument("--quick", action="store_true", help="Run a tiny smoke version.")
    p.add_argument("--train-pairs", type=int, default=1200)
    p.add_argument("--test-pairs", type=int, default=800)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-3, 1e-2, 1e-1, 5e-1, 1.0])
    p.add_argument("--bon-prompts", type=int, default=400)
    p.add_argument("--n-values", type=int, nargs="+", default=[1, 4, 16, 64])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.train_pairs = 500
        args.test_pairs = 300
        args.epochs = 35
        args.seeds = [0]
        args.weight_decays = [0.0, 1e-1, 5e-1]
        args.bon_prompts = 120

    cfg = WorkflowConfig(
        output_dir=args.output_dir,
        train_pairs=args.train_pairs,
        test_pairs=args.test_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=tuple(args.seeds),
        weight_decays=tuple(args.weight_decays),
        n_values=tuple(args.n_values),
        bon_prompts=args.bon_prompts,
        quick=args.quick,
    )
    run_workflow(cfg)
    print(f"Wrote group-meeting workflow results to {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()
