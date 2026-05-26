#!/usr/bin/env python3
"""
Small synthetic experiment: weight decay vs reward hacking.

The experiment creates a reward-modeling dataset where "true" features determine
oracle preferences, while "spurious" features are correlated with true quality
only in the training distribution. A reward model trained on this data can learn
spurious shortcuts. We scan weight decay and test whether it reduces shortcut
reliance and improves robustness under best-of-N optimization pressure.

Run:
    python src/run_experiment.py --quick
    python src/run_experiment.py --train-pairs 50000 --epochs 12 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------


@dataclass
class DataConfig:
    true_dim: int = 8
    spur_dim: int = 8
    noise_dim: int = 16
    train_rho: float = 0.90
    iid_rho: float = 0.90
    ood_rho: float = 0.00
    reverse_rho: float = -0.90
    label_noise_std: float = 0.10
    feature_noise_std: float = 0.05
    true_observation_noise_std: float = 0.75
    conflict_strength: float = 2.00

    @property
    def x_dim(self) -> int:
        return self.true_dim + self.spur_dim + self.noise_dim


@dataclass
class TrainConfig:
    train_pairs: int = 30_000
    test_pairs: int = 8_000
    batch_size: int = 512
    epochs: int = 8
    lr: float = 2e-3
    bottleneck_dim: int = 4
    hidden_dim: int = 64
    dropout: float = 0.0
    weight_decays: Tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    seeds: Tuple[int, ...] = (0, 1, 2)
    n_values: Tuple[int, ...] = (1, 4, 16, 64, 256)
    bon_prompts: int = 2_000
    device: str = "cpu"
    output_dir: str = "results"


# -----------------------------
# Reproducibility and utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_logit_loss(chosen_reward: torch.Tensor, rejected_reward: torch.Tensor) -> torch.Tensor:
    # Bradley-Terry pairwise loss: -log(sigmoid(r_chosen - r_rejected))
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


# -----------------------------
# Synthetic data generator
# -----------------------------


class SyntheticPreferenceWorld:
    """Known true and spurious latent features.

    A response vector x is concatenated as:
        x = [true_features, spurious_features, nuisance_noise]

    Oracle utility depends only on true_features. The training distribution makes
    spurious features correlated with oracle utility, which creates a shortcut.
    """

    def __init__(self, cfg: DataConfig, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed + 12345)

        true_w = torch.randn(cfg.true_dim, generator=g)
        spur_w = torch.randn(cfg.spur_dim, generator=g)
        self.true_w = true_w / true_w.norm().clamp_min(1e-12)
        self.spur_w = spur_w / spur_w.norm().clamp_min(1e-12)

    def _make_candidates(
        self,
        n: int,
        rho: float,
        generator: torch.Generator,
        conflict: bool = False,
        hack_distribution: bool = False,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        z_true_clean = torch.randn(n, cfg.true_dim, generator=generator)
        true_u = z_true_clean @ self.true_w
        z_true = z_true_clean + cfg.true_observation_noise_std * torch.randn(n, cfg.true_dim, generator=generator)
        true_u = true_u + cfg.label_noise_std * torch.randn(n, generator=generator)

        eps_spur = torch.randn(n, cfg.spur_dim, generator=generator)
        true_component = true_u[:, None] * self.spur_w[None, :]
        rho_abs = min(abs(rho), 0.999)
        spur = rho * true_component + math.sqrt(max(1.0 - rho_abs**2, 0.0)) * eps_spur

        if conflict:
            # Make spurious utility intentionally disagree with true utility.
            spur = -cfg.conflict_strength * true_u[:, None] * self.spur_w[None, :] + 0.25 * eps_spur

        if hack_distribution:
            # A distribution where very high spurious style is available, while
            # true quality is not improved and can even be anti-correlated.
            # This mimics an optimizer discovering style-only high reward regions.
            style_scale = torch.distributions.Exponential(rate=torch.tensor(1.0)).sample((n,))
            style_sign = torch.ones(n)
            spur = (
                cfg.conflict_strength * style_scale[:, None] * style_sign[:, None] * self.spur_w[None, :]
                - 0.35 * true_u[:, None] * self.spur_w[None, :]
                + 0.25 * eps_spur
            )

        spur = spur + cfg.feature_noise_std * torch.randn(n, cfg.spur_dim, generator=generator)
        nuisance = torch.randn(n, cfg.noise_dim, generator=generator)

        x = torch.cat([z_true, spur, nuisance], dim=1).float()
        spur_u = (spur @ self.spur_w).float()

        return {
            "x": x,
            "true_u": true_u.float(),
            "spur_u": spur_u.float(),
        }

    def make_pair_dataset(
        self,
        n_pairs: int,
        rho: float,
        seed: int,
        conflict: bool = False,
    ) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        a = self._make_candidates(n_pairs, rho=rho, generator=g, conflict=conflict)
        b = self._make_candidates(n_pairs, rho=rho, generator=g, conflict=conflict)

        choose_a = a["true_u"] >= b["true_u"]
        x_chosen = torch.where(choose_a[:, None], a["x"], b["x"])
        x_rejected = torch.where(choose_a[:, None], b["x"], a["x"])
        chosen_true = torch.where(choose_a, a["true_u"], b["true_u"])
        rejected_true = torch.where(choose_a, b["true_u"], a["true_u"])
        chosen_spur = torch.where(choose_a, a["spur_u"], b["spur_u"])
        rejected_spur = torch.where(choose_a, b["spur_u"], a["spur_u"])

        return {
            "x_chosen": x_chosen,
            "x_rejected": x_rejected,
            "chosen_true": chosen_true,
            "rejected_true": rejected_true,
            "chosen_spur": chosen_spur,
            "rejected_spur": rejected_spur,
        }

    def make_candidate_pool(
        self,
        n_prompts: int,
        n_candidates: int,
        seed: int,
        hack_distribution: bool = True,
    ) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        n = n_prompts * n_candidates
        cand = self._make_candidates(
            n,
            rho=self.cfg.reverse_rho,
            generator=g,
            conflict=False,
            hack_distribution=hack_distribution,
        )
        return {
            "x": cand["x"].reshape(n_prompts, n_candidates, -1),
            "true_u": cand["true_u"].reshape(n_prompts, n_candidates),
            "spur_u": cand["spur_u"].reshape(n_prompts, n_candidates),
        }

    def make_sensitivity_batch(self, n: int, seed: int) -> Dict[str, torch.Tensor]:
        """Create paired interventions: same true/noise, low vs high spurious style."""
        g = torch.Generator().manual_seed(seed)
        cfg = self.cfg
        z_true = torch.randn(n, cfg.true_dim, generator=g)
        true_u = z_true @ self.true_w
        nuisance = torch.randn(n, cfg.noise_dim, generator=g)

        low_spur = -cfg.conflict_strength * self.spur_w[None, :].repeat(n, 1)
        high_spur = cfg.conflict_strength * self.spur_w[None, :].repeat(n, 1)
        neutral_spur = torch.zeros(n, cfg.spur_dim)

        x_low_spur = torch.cat([z_true, low_spur, nuisance], dim=1).float()
        x_high_spur = torch.cat([z_true, high_spur, nuisance], dim=1).float()

        # True intervention: same spurious/noise, low vs high true quality.
        true_dir = self.true_w[None, :].repeat(n, 1)
        low_true = -cfg.conflict_strength * true_dir
        high_true = cfg.conflict_strength * true_dir
        x_low_true = torch.cat([low_true, neutral_spur, nuisance], dim=1).float()
        x_high_true = torch.cat([high_true, neutral_spur, nuisance], dim=1).float()

        return {
            "x_low_spur": x_low_spur,
            "x_high_spur": x_high_spur,
            "x_low_true": x_low_true,
            "x_high_true": x_high_true,
            "true_u": true_u.float(),
        }


# -----------------------------
# Model
# -----------------------------


class RewardModel(nn.Module):
    """Small reward model with a representation bottleneck.

    The bottleneck makes the setting closer to a superposition-like compression
    problem: more latent causes than explicit representation dimensions.
    """

    def __init__(self, x_dim: int, hidden_dim: int, bottleneck_dim: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(bottleneck_dim, 1, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.encode(x)).squeeze(-1)


# -----------------------------
# Training and evaluation
# -----------------------------


def train_one_model(
    model: RewardModel,
    train_data: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    weight_decay: float,
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.to(device)

    ds = TensorDataset(train_data["x_chosen"], train_data["x_rejected"])
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    history: Dict[str, float] = {}
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for x_c, x_r in dl:
            x_c = x_c.to(device)
            x_r = x_r.to(device)
            r_c = model(x_c)
            r_r = model(x_r)
            loss = safe_logit_loss(r_c, r_r)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history[f"train_loss_epoch_{epoch + 1}"] = float(np.mean(losses))
    history["train_loss_final"] = history[f"train_loss_epoch_{cfg.epochs}"]
    return history


@torch.no_grad()
def evaluate_pairs(model: RewardModel, data: Dict[str, torch.Tensor], device: str) -> Dict[str, float]:
    model.eval()
    x_c = data["x_chosen"].to(device)
    x_r = data["x_rejected"].to(device)
    r_c = model(x_c).cpu()
    r_r = model(x_r).cpu()
    margin = r_c - r_r
    acc = (margin > 0).float().mean().item()
    return {
        "accuracy": acc,
        "mean_margin": float(margin.mean()),
        "median_margin": float(margin.median()),
    }


@torch.no_grad()
def evaluate_candidate_correlations(
    model: RewardModel,
    world: SyntheticPreferenceWorld,
    n: int,
    rho: float,
    seed: int,
    device: str,
    conflict: bool = False,
) -> Dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    cand = world._make_candidates(n, rho=rho, generator=g, conflict=conflict)
    x = cand["x"].to(device)
    reward = model(x).cpu()
    true_u = cand["true_u"].cpu()
    spur_u = cand["spur_u"].cpu()
    return {
        "reward_true_corr": pearson_corr(reward, true_u),
        "reward_spur_corr": pearson_corr(reward, spur_u),
    }


@torch.no_grad()
def evaluate_sensitivity(
    model: RewardModel,
    world: SyntheticPreferenceWorld,
    seed: int,
    device: str,
    n: int = 8_000,
) -> Dict[str, float]:
    model.eval()
    batch = world.make_sensitivity_batch(n=n, seed=seed)
    out = {}
    for k, v in batch.items():
        if k.startswith("x_"):
            out[k] = model(v.to(device)).cpu()

    spur_delta = out["x_high_spur"] - out["x_low_spur"]
    true_delta = out["x_high_true"] - out["x_low_true"]
    return {
        "spurious_sensitivity": float(spur_delta.mean()),
        "true_sensitivity": float(true_delta.mean()),
        "abs_spurious_over_true_sensitivity": float(
            spur_delta.abs().mean() / true_delta.abs().mean().clamp_min(1e-12)
        ),
    }


def ridge_probe_direction(h: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Closed-form ridge regression direction mapping h -> y."""
    h = h.detach().float()
    y = y.detach().float().flatten()
    h = h - h.mean(dim=0, keepdim=True)
    y = y - y.mean()
    xtx = h.T @ h
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
    xty = h.T @ y
    w = torch.linalg.solve(xtx + ridge * eye, xty)
    return w


@torch.no_grad()
def evaluate_representation_mixing(
    model: RewardModel,
    world: SyntheticPreferenceWorld,
    n: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    """Probe-based proxy for representation mixing.

    We encode a candidate batch, fit linear probes for true utility and spurious
    utility on the bottleneck representation, then compute the absolute cosine
    between the two probe directions. High cosine means the representation makes
    true and spurious utility less separable. This is not a direct measurement of
    superposition, but it is a useful cheap proxy in this controlled setup.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    cand = world._make_candidates(n, rho=0.0, generator=g, conflict=False)
    x = cand["x"].to(device)
    h = model.encode(x).cpu()
    true_u = cand["true_u"].cpu()
    spur_u = cand["spur_u"].cpu()

    w_true = ridge_probe_direction(h, true_u)
    w_spur = ridge_probe_direction(h, spur_u)
    cos = torch.dot(w_true, w_spur) / (w_true.norm() * w_spur.norm()).clamp_min(1e-12)

    # Probe R^2 values tell us whether the bottleneck still carries each signal.
    def r2_score(w: torch.Tensor, y: torch.Tensor) -> float:
        pred = (h - h.mean(dim=0, keepdim=True)) @ w + y.mean()
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-12)
        return float(1.0 - ss_res / ss_tot)

    return {
        "probe_abs_cos_true_spur": float(cos.abs()),
        "probe_r2_true": r2_score(w_true, true_u),
        "probe_r2_spur": r2_score(w_spur, spur_u),
        "bottleneck_l2_mean": float(h.norm(dim=1).mean()),
    }


@torch.no_grad()
def evaluate_best_of_n(
    model: RewardModel,
    world: SyntheticPreferenceWorld,
    cfg: TrainConfig,
    weight_decay: float,
    seed: int,
) -> List[Dict[str, float]]:
    """Best-of-N simulation of optimization pressure.

    For each prompt we create a pool of max(N) candidates from a distribution
    where high spurious style is easy to find. The reward model selects the
    highest proxy-reward candidate among the first N candidates. We evaluate the
    chosen candidate with oracle true utility.
    """
    model.eval()
    device = torch.device(cfg.device)
    n_max = max(cfg.n_values)
    pool = world.make_candidate_pool(
        n_prompts=cfg.bon_prompts,
        n_candidates=n_max,
        seed=seed,
        hack_distribution=True,
    )
    x = pool["x"].to(device)
    flat_x = x.reshape(-1, x.shape[-1])
    rewards = model(flat_x).reshape(cfg.bon_prompts, n_max).cpu()
    true_u = pool["true_u"].cpu()
    spur_u = pool["spur_u"].cpu()

    rows = []
    for n in cfg.n_values:
        r_n = rewards[:, :n]
        idx = r_n.argmax(dim=1)
        row_idx = torch.arange(cfg.bon_prompts)
        selected_true = true_u[row_idx, idx]
        selected_spur = spur_u[row_idx, idx]
        selected_reward = r_n[row_idx, idx]

        oracle_idx = true_u[:, :n].argmax(dim=1)
        oracle_true = true_u[row_idx, oracle_idx]
        random_true = true_u[:, 0]

        rows.append(
            {
                "seed": seed,
                "weight_decay": weight_decay,
                "N": n,
                "selected_proxy_reward_mean": float(selected_reward.mean()),
                "selected_true_utility_mean": float(selected_true.mean()),
                "selected_spurious_utility_mean": float(selected_spur.mean()),
                "oracle_best_true_utility_mean": float(oracle_true.mean()),
                "random_candidate_true_utility_mean": float(random_true.mean()),
                "proxy_minus_true_gap": float(selected_reward.mean() - selected_true.mean()),
            }
        )
    return rows


# -----------------------------
# Plotting
# -----------------------------


def plot_with_errorbars(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str | None,
    title: str,
    ylabel: str,
    outpath: Path,
    xscale_log: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    if group is None:
        agg = df.groupby(x)[y].agg(["mean", "std"]).reset_index()
        plt.errorbar(agg[x], agg["mean"], yerr=agg["std"], marker="o", capsize=3)
    else:
        for name, sub in df.groupby(group):
            agg = sub.groupby(x)[y].agg(["mean", "std"]).reset_index()
            plt.errorbar(agg[x], agg["mean"], yerr=agg["std"], marker="o", capsize=3, label=str(name))
        plt.legend(title=group)
    if xscale_log:
        plt.xscale("symlog", linthresh=1e-6)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def make_plots(summary_df: pd.DataFrame, bon_df: pd.DataFrame, outdir: Path) -> None:
    plot_with_errorbars(
        summary_df,
        x="weight_decay",
        y="reverse_accuracy",
        group=None,
        title="OOD reverse preference accuracy vs weight decay",
        ylabel="reverse accuracy",
        outpath=outdir / "reverse_accuracy_vs_weight_decay.png",
        xscale_log=True,
    )
    plot_with_errorbars(
        summary_df,
        x="weight_decay",
        y="conflict_accuracy",
        group=None,
        title="Conflict-set accuracy vs weight decay",
        ylabel="conflict accuracy",
        outpath=outdir / "conflict_accuracy_vs_weight_decay.png",
        xscale_log=True,
    )
    plot_with_errorbars(
        summary_df,
        x="weight_decay",
        y="abs_spurious_over_true_sensitivity",
        group=None,
        title="Spurious / true sensitivity ratio vs weight decay",
        ylabel="abs spurious sensitivity / true sensitivity",
        outpath=outdir / "sensitivity_ratio_vs_weight_decay.png",
        xscale_log=True,
    )
    plot_with_errorbars(
        summary_df,
        x="weight_decay",
        y="probe_abs_cos_true_spur",
        group=None,
        title="Probe mixing proxy vs weight decay",
        ylabel="abs cosine between true and spurious probes",
        outpath=outdir / "probe_mixing_vs_weight_decay.png",
        xscale_log=True,
    )

    # Best-of-N curves: each weight decay is a separate line.
    for metric, label, filename in [
        (
            "selected_true_utility_mean",
            "selected oracle true utility",
            "best_of_n_selected_true_utility.png",
        ),
        (
            "selected_proxy_reward_mean",
            "selected proxy reward",
            "best_of_n_proxy_reward.png",
        ),
        (
            "selected_spurious_utility_mean",
            "selected spurious utility",
            "best_of_n_spurious_utility.png",
        ),
    ]:
        plt.figure(figsize=(8, 5))
        for wd, sub in bon_df.groupby("weight_decay"):
            agg = sub.groupby("N")[metric].agg(["mean", "std"]).reset_index()
            plt.errorbar(agg["N"], agg["mean"], yerr=agg["std"], marker="o", capsize=3, label=f"wd={wd:g}")
        plt.xscale("log", base=2)
        plt.title(label + " under best-of-N")
        plt.xlabel("N candidates")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()


# -----------------------------
# Main
# -----------------------------


def run_experiment(cfg: TrainConfig, data_cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir = ensure_dir(cfg.output_dir)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"train": asdict(cfg), "data": asdict(data_cfg)}, f, indent=2)

    summary_rows: List[Dict[str, float]] = []
    bon_rows: List[Dict[str, float]] = []

    for seed in cfg.seeds:
        set_seed(seed)
        world = SyntheticPreferenceWorld(data_cfg, seed=seed)

        train_data = world.make_pair_dataset(
            n_pairs=cfg.train_pairs,
            rho=data_cfg.train_rho,
            seed=10_000 + seed,
            conflict=False,
        )
        iid_data = world.make_pair_dataset(
            n_pairs=cfg.test_pairs,
            rho=data_cfg.iid_rho,
            seed=20_000 + seed,
            conflict=False,
        )
        ood_data = world.make_pair_dataset(
            n_pairs=cfg.test_pairs,
            rho=data_cfg.ood_rho,
            seed=30_000 + seed,
            conflict=False,
        )
        reverse_data = world.make_pair_dataset(
            n_pairs=cfg.test_pairs,
            rho=data_cfg.reverse_rho,
            seed=40_000 + seed,
            conflict=False,
        )
        conflict_data = world.make_pair_dataset(
            n_pairs=cfg.test_pairs,
            rho=data_cfg.reverse_rho,
            seed=50_000 + seed,
            conflict=True,
        )

        for wd in tqdm(cfg.weight_decays, desc=f"seed={seed}"):
            set_seed(seed)
            model = RewardModel(
                x_dim=data_cfg.x_dim,
                hidden_dim=cfg.hidden_dim,
                bottleneck_dim=cfg.bottleneck_dim,
                dropout=cfg.dropout,
            )

            train_hist = train_one_model(model, train_data, cfg, weight_decay=wd)
            iid = evaluate_pairs(model, iid_data, cfg.device)
            ood = evaluate_pairs(model, ood_data, cfg.device)
            reverse = evaluate_pairs(model, reverse_data, cfg.device)
            conflict = evaluate_pairs(model, conflict_data, cfg.device)
            sens = evaluate_sensitivity(model, world, seed=60_000 + seed, device=cfg.device)
            corr = evaluate_candidate_correlations(
                model,
                world,
                n=cfg.test_pairs,
                rho=0.0,
                seed=70_000 + seed,
                device=cfg.device,
            )
            mix = evaluate_representation_mixing(
                model,
                world,
                n=cfg.test_pairs,
                seed=80_000 + seed,
                device=cfg.device,
            )
            bon_rows.extend(
                evaluate_best_of_n(
                    model,
                    world,
                    cfg,
                    weight_decay=wd,
                    seed=90_000 + seed,
                )
            )

            row: Dict[str, float] = {
                "seed": seed,
                "weight_decay": wd,
                "train_loss_final": train_hist["train_loss_final"],
                "iid_accuracy": iid["accuracy"],
                "iid_mean_margin": iid["mean_margin"],
                "ood_accuracy": ood["accuracy"],
                "ood_mean_margin": ood["mean_margin"],
                "reverse_accuracy": reverse["accuracy"],
                "reverse_mean_margin": reverse["mean_margin"],
                "conflict_accuracy": conflict["accuracy"],
                "conflict_mean_margin": conflict["mean_margin"],
                **sens,
                **corr,
                **mix,
            }
            summary_rows.append(row)

            # Save incrementally in case the run is interrupted.
            pd.DataFrame(summary_rows).to_csv(outdir / "summary_metrics.csv", index=False)
            pd.DataFrame(bon_rows).to_csv(outdir / "best_of_n_metrics.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    bon_df = pd.DataFrame(bon_rows)

    # Aggregate tables.
    summary_agg = summary_df.groupby("weight_decay").agg(["mean", "std"])
    summary_agg.columns = ["_".join([str(c) for c in col if c]) for col in summary_agg.columns.values]
    summary_agg.reset_index().to_csv(outdir / "summary_by_weight_decay.csv", index=False)

    bon_agg = bon_df.groupby(["weight_decay", "N"]).agg(["mean", "std"])
    bon_agg.columns = ["_".join([str(c) for c in col if c]) for col in bon_agg.columns.values]
    bon_agg.reset_index().to_csv(outdir / "best_of_n_by_weight_decay.csv", index=False)

    make_plots(summary_df, bon_df, outdir)
    return summary_df, bon_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight decay vs reward hacking synthetic experiment")
    p.add_argument("--quick", action="store_true", help="Run a small smoke-test configuration")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--train-pairs", type=int, default=30_000)
    p.add_argument("--test-pairs", type=int, default=8_000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--bottleneck-dim", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument("--n-values", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    p.add_argument("--bon-prompts", type=int, default=2_000)
    p.add_argument("--train-rho", type=float, default=0.90)
    p.add_argument("--reverse-rho", type=float, default=-0.90)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.train_pairs = 2_000
        args.test_pairs = 800
        args.epochs = 2
        args.seeds = [0]
        args.weight_decays = [0.0, 1e-2, 1.0]
        args.n_values = [1, 4, 16, 64]
        args.bon_prompts = 200

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    data_cfg = DataConfig(train_rho=args.train_rho, reverse_rho=args.reverse_rho)
    cfg = TrainConfig(
        train_pairs=args.train_pairs,
        test_pairs=args.test_pairs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decays=tuple(args.weight_decays),
        seeds=tuple(args.seeds),
        n_values=tuple(args.n_values),
        bon_prompts=args.bon_prompts,
        device=args.device,
        output_dir=args.output_dir,
    )

    summary_df, bon_df = run_experiment(cfg, data_cfg)

    print("\nDone. Key mean metrics by weight_decay:\n")
    key_cols = [
        "iid_accuracy",
        "reverse_accuracy",
        "conflict_accuracy",
        "abs_spurious_over_true_sensitivity",
        "probe_abs_cos_true_spur",
        "reward_true_corr",
        "reward_spur_corr",
    ]
    display = summary_df.groupby("weight_decay")[key_cols].mean().reset_index()
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(display)

    print(f"\nWrote results to: {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()
