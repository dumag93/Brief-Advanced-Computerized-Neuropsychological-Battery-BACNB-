#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated BACNB Monte Carlo simulation.

Objective:
    Simulate the same agents across the four BACNB tests (SART, SST, Flanker, and
    Digit Span), extract profiles/clusters from observable data, and compare
    the emergent constructs with the Miyake/Friedman theoretical model.

The script is deliberately transparent:
    1. Generates continuous latent microparameters per agent.
    2. Converts these microparameters into observable metrics per task.
    3. Standardizes metrics in a single deficit-oriented direction.
    4. Clusters variables to identify emergent constructs.
    5. Clusters agents to identify functional profiles.
    6. Only then compares the emergent constructs with Miyake.

This avoids circularity: Miyake does not define the emergent clusters.

Usage:
    python simulation/Monte_Carlo_Simulation_BACNB.py --n 200000 --seed 20260610 --noise 1.0

Adjustable parameters:
    --n             synthetic sample size
    --seed          reproducible seed
    --noise         global observational-noise multiplier
    --max-k         maximum K tested for agent clustering
    --sample-k      subsample used for K selection
    --outdir        optional output folder
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.special import expit
from scipy import stats


ROOT = Path(__file__).resolve().parent
DATE_TAG = datetime.now().strftime("%Y-%m-%d")
DEFAULT_SEED = 20260610
DEFAULT_N = 200000
Z_CLIP = 4.0
ALPHA = 0.05


@dataclass(frozen=True)
class FeatureSpec:
    col: str
    label: str
    short: str
    test: str
    higher_worse: bool
    miyake_inhibition: float = 0.0
    miyake_working_memory: float = 0.0
    miyake_flexibility: float = 0.0


FEATURES: list[FeatureSpec] = [
    FeatureSpec("sart_commission_rate", "SART No-Go commissions", "SART com.", "SART", True, 1.00, 0.00, 0.10),
    FeatureSpec("sart_omission_rate", "SART Go omissions", "SART om.", "SART", True, 0.20, 0.45, 0.05),
    FeatureSpec("sart_artifact_rate", "SART anticipations", "SART ant.", "SART", True, 0.75, 0.00, 0.10),
    FeatureSpec("sart_mean_rt", "SART mean RT", "SART RT", "SART", True, 0.10, 0.10, 0.05),
    FeatureSpec("sart_cv_rt", "SART RT variability", "SART CV", "SART", True, 0.25, 0.45, 0.10),
    FeatureSpec("sart_fatigue_delta", "SART post-fatigue delta", "SART fad.", "SART", True, 0.20, 0.35, 0.20),
    FeatureSpec("sart_automation_index", "SART automation/rigidity", "SART auto.", "SART", True, 0.75, 0.10, 0.35),
    FeatureSpec("sst_p_respond_signal", "SST p(respond|signal)", "SST pResp", "SST", True, 1.00, 0.00, 0.05),
    FeatureSpec("sst_ssrt", "SST estimated SSRT", "SST SSRT", "SST", True, 1.00, 0.05, 0.00),
    FeatureSpec("sst_go_rt", "SST mean Go RT", "SST RT", "SST", True, 0.10, 0.10, 0.00),
    FeatureSpec("sst_go_omission_rate", "SST Go omissions", "SST om.", "SST", True, 0.20, 0.35, 0.05),
    FeatureSpec("sst_go_choice_error_rate", "SST Go choice error", "SST choice", "SST", True, 0.35, 0.20, 0.05),
    FeatureSpec("sst_tracking_error", "SST SSD tracking error", "SST track", "SST", True, 0.30, 0.25, 0.15),
    FeatureSpec("flanker_effect", "Flanker incongruent effect", "FL effect", "FLANKER", True, 0.70, 0.10, 0.10),
    FeatureSpec("flanker_interference", "Flanker interference", "FL interf.", "FLANKER", True, 0.70, 0.10, 0.10),
    FeatureSpec("flanker_incongruent_error", "Flanker incongruent error", "FL err inc.", "FLANKER", True, 0.65, 0.10, 0.05),
    FeatureSpec("flanker_accuracy", "Flanker total accuracy", "FL acc.", "FLANKER", False, 0.55, 0.10, 0.05),
    FeatureSpec("flanker_mean_rt", "Flanker mean RT", "FL RT", "FLANKER", True, 0.10, 0.10, 0.00),
    FeatureSpec("flanker_cv_rt", "Flanker RT variability", "FL CV", "FLANKER", True, 0.20, 0.35, 0.05),
    FeatureSpec("ds_forward_span", "DS forward span", "DS dir.", "DS", False, 0.00, 0.70, 0.00),
    FeatureSpec("ds_backward_span", "DS backward span", "DS inv.", "DS", False, 0.10, 1.00, 0.05),
    FeatureSpec("ds_span_sum", "DS span sum", "DS sum", "DS", False, 0.05, 1.00, 0.00),
    FeatureSpec("ds_backward_cost", "DS backward cost", "DS cost", "DS", True, 0.15, 0.75, 0.05),
    FeatureSpec("ds_accuracy", "DS total accuracy", "DS acc.", "DS", False, 0.05, 0.90, 0.05),
    FeatureSpec("ds_response_time", "DS response time", "DS time", "DS", True, 0.05, 0.25, 0.05),
]


TEST_LABELS = {
    "SART": "SART",
    "SST": "SST",
    "FLANKER": "Flanker",
    "DS": "Digit Span",
}

MIYAKE_LABELS = {
    "inhibition": "Inhibitory Control",
    "working_memory": "Working Memory",
    "flexibility": "Cognitive Flexibility",
}

LATENT_LABELS = {
    "general_control_deficit": "General executive factor",
    "slowness": "Slowness/conservative strategy",
    "motor_impulsivity": "Motor impulsivity",
    "stop_deficit": "Stop-cancellation cost",
    "attention_lapse": "Attentional lapses",
    "temporal_variability": "Temporal variability",
    "conflict_susceptibility": "Conflict susceptibility",
    "verbal_wm_deficit": "Verbal-memory cost",
    "rigidity_automation": "Rigidity/automation",
}

PROFILE_NAMES = {
    1: "Flexible regulation",
    2: "Exploratory variability",
    3: "Prepotent automation",
    4: "Global control economy",
}

PROFILE_SHORT = {
    1: "P1",
    2: "P2",
    3: "P3",
    4: "P4",
}

PROFILE_SLUGS = {
    1: "P1_preserved_control",
    2: "P2_lapses_variability",
    3: "P3_prepotent_automation",
    4: "P4_broad_deficit",
}

PROFILE_COLUMNS = {
    1: "sim_P1_preserved_control",
    2: "sim_P2_lapses_variability",
    3: "sim_P3_prepotent_automation",
    4: "sim_P4_broad_deficit",
}

DISTANCE_COLUMNS = {
    1: "dist_P1_preserved_control",
    2: "dist_P2_lapses_variability",
    3: "dist_P3_prepotent_automation",
    4: "dist_P4_broad_deficit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated BACNB Monte Carlo")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Synthetic sample size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--noise", type=float, default=1.0, help="Global observational-noise multiplier")
    parser.add_argument("--max-k", type=int, default=8, help="Maximum K for agent clustering")
    parser.add_argument("--sample-k", type=int, default=20000, help="Subsample for K selection")
    parser.add_argument("--outdir", type=str, default="", help="Optional output folder")
    parser.add_argument("--similarity-tau", type=float, default=1.0, help="Functional-similarity softmax temperature")
    parser.add_argument("--similarity-method", choices=["mahalanobis", "euclidean"], default="mahalanobis", help="Metric used for prototype distance")
    parser.add_argument("--similarity-space", choices=["metrics", "constructs", "both"], default="metrics", help="Space used for functional similarity")
    parser.add_argument("--run-synthetic-cases", action="store_true", default=True, help="Generate synthetic validation cases")
    parser.add_argument("--no-synthetic-cases", action="store_true", help="Disable synthetic validation cases")
    parser.add_argument("--individual-file", type=str, default="", help="Optional CSV/JSON file with individual observed metrics")
    parser.add_argument("--z-reference-file", type=str, default="", help="Optional JSON file with oriented functional-z reference")
    parser.add_argument("--state-reference-file", type=str, default="", help="Optional JSON file with state-space reference")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_style() -> None:
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 2.0,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
        }
    )


def max_standard_error_proportion(n: int) -> float:
    """Maximum standard error for proportions at p = 0.5."""
    return math.sqrt(0.25 / max(n, 1))


def format_p_apa(p: float) -> str:
    """Formats a p-value in the APA style used in the report."""
    if not np.isfinite(p):
        return "p = N/A"
    if p < 0.001:
        return "p < .001"
    txt = f"{p:.3f}"
    if txt.startswith("0"):
        txt = txt[1:]
    return f"p = {txt}"


def zstandardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-9:
        sd = np.nanstd(x)
        return (x - np.nanmean(x)) / (sd + 1e-9)
    return 0.67448975 * (x - med) / (mad + 1e-9)


def clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def bounded_int(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.rint(np.clip(x, lo, hi)).astype(int)


def sample_latents(n: int, rng: np.random.Generator, noise: float) -> pd.DataFrame:
    """Generates continuous microparameters.

    All signals are deficit-oriented: higher values indicate greater
    difficulty/noise in that microprocess.
    """

    # The generator uses continuous functional heterogeneity through sparse mixing.
    # This avoids a purely Gaussian population that would collapse into "low vs.
    # high deficit" while also avoiding preassigned subtype labels
    # to the algorithm. Clustering sees only observable metrics.
    archetypes = np.array(
        [
            [-0.85, -0.15, -0.45, -0.45, -0.35, -0.35, -0.35, -0.35, -0.45],  # efficient
            [0.35, -0.55, 1.45, 0.70, 0.00, 0.20, 0.40, 0.10, 1.65],       # impulsive/automated
            [0.55, 0.25, 0.10, 0.20, 1.55, 1.50, 0.25, 0.55, 0.25],        # lapse/variable
            [0.20, 1.55, -0.65, 0.20, 0.20, 0.45, 0.10, 0.30, 0.00],       # slow/conservative
            [0.35, 0.10, 0.50, 1.75, 0.20, 0.25, 0.20, 0.00, 0.20],        # cancellation
            [0.30, 0.00, 0.45, 0.20, 0.20, 0.25, 1.70, 0.10, 0.45],        # conflict
            [0.30, 0.20, 0.00, 0.00, 0.40, 0.20, 0.10, 1.75, 0.10],        # verbal memory
            [1.00, 0.60, 0.80, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90],        # mixed
        ],
        dtype=float,
    )
    n_arch = archetypes.shape[0]
    dominant = rng.choice(n_arch, size=n, p=np.array([0.18, 0.14, 0.14, 0.12, 0.13, 0.13, 0.11, 0.05]))
    alpha = np.full(n_arch, 0.18)
    weights = np.empty((n, n_arch), dtype=float)
    for i, d in enumerate(dominant):
        a = alpha.copy()
        a[d] += 2.8
        weights[i] = rng.dirichlet(a)
    severity = rng.lognormal(mean=0.02, sigma=0.32, size=n)
    style = (weights @ archetypes) * severity[:, None]

    base = rng.normal(0, 0.55, size=(n, 9))
    general_seed = rng.normal(0, 0.55, n)
    heavy_tail = rng.standard_t(df=5, size=(n, 3)) * 0.08 * noise

    general = zstandardize(style[:, 0] + base[:, 0] + 0.25 * general_seed + heavy_tail[:, 0])
    slowness = zstandardize(style[:, 1] + base[:, 1] + 0.25 * general - 0.20 * style[:, 2])
    impulsivity = zstandardize(style[:, 2] + base[:, 2] + 0.25 * general - 0.25 * slowness)
    stop = zstandardize(style[:, 3] + base[:, 3] + 0.30 * general + 0.25 * impulsivity)
    attention = zstandardize(style[:, 4] + base[:, 4] + 0.35 * general + heavy_tail[:, 1])
    variability = zstandardize(style[:, 5] + base[:, 5] + 0.35 * attention + 0.20 * general)
    conflict = zstandardize(style[:, 6] + base[:, 6] + 0.25 * general + 0.25 * impulsivity)
    wm = zstandardize(style[:, 7] + base[:, 7] + 0.25 * attention + 0.25 * general)
    rigidity = zstandardize(style[:, 8] + base[:, 8] + 0.25 * conflict + 0.30 * impulsivity + heavy_tail[:, 2])

    return pd.DataFrame(
        {
            "general_control_deficit": general,
            "slowness": slowness,
            "motor_impulsivity": impulsivity,
            "stop_deficit": stop,
            "attention_lapse": attention,
            "temporal_variability": variability,
            "conflict_susceptibility": conflict,
            "verbal_wm_deficit": wm,
            "rigidity_automation": rigidity,
        }
    )


def simulate_sart(lat: pd.DataFrame, rng: np.random.Generator, noise: float) -> pd.DataFrame:
    n = len(lat)
    g = lat["general_control_deficit"].to_numpy()
    slow = lat["slowness"].to_numpy()
    imp = lat["motor_impulsivity"].to_numpy()
    stop = lat["stop_deficit"].to_numpy()
    att = lat["attention_lapse"].to_numpy()
    var = lat["temporal_variability"].to_numpy()
    rigid = lat["rigidity_automation"].to_numpy()

    p_comm = expit(-2.25 + 0.80 * imp + 0.65 * rigid + 0.30 * stop + 0.20 * g + rng.normal(0, 0.25 * noise, n))
    p_omiss = expit(-4.20 + 0.85 * att + 0.35 * slow + 0.30 * var + 0.20 * g + rng.normal(0, 0.20 * noise, n))
    p_art = expit(-4.30 + 1.05 * imp + 0.55 * rigid - 0.35 * slow + rng.normal(0, 0.25 * noise, n))

    comm = rng.binomial(15, clip(p_comm, 0.001, 0.999))
    omiss = rng.binomial(120, clip(p_omiss, 0.001, 0.999))
    art = rng.binomial(135, clip(p_art, 0.001, 0.999))

    mean_rt = clip(265 + 52 * slow - 42 * imp + 14 * g + rng.normal(0, 24 * noise, n), 105, 700)
    cv_rt = clip(18 + 7.5 * var + 3.5 * att + rng.normal(0, 4 * noise, n), 4, 90)
    fatigue_delta = clip(2.0 + 4.0 * att + 2.4 * rigid + 1.0 * var + rng.normal(0, 3.0 * noise, n), -15, 45)

    commission_rate = 100 * comm / 15
    omission_rate = 100 * omiss / 120
    artifact_rate = 100 * art / 135
    automation_index = clip(0.45 * commission_rate + 0.30 * artifact_rate + 8.0 * zstandardize(rigid) + 5.0 * zstandardize(imp), 0, 120)

    return pd.DataFrame(
        {
            "sart_commission_rate": commission_rate,
            "sart_omission_rate": omission_rate,
            "sart_artifact_rate": artifact_rate,
            "sart_mean_rt": mean_rt,
            "sart_cv_rt": cv_rt,
            "sart_fatigue_delta": fatigue_delta,
            "sart_automation_index": automation_index,
        }
    )


def simulate_sst(lat: pd.DataFrame, rng: np.random.Generator, noise: float) -> pd.DataFrame:
    n = len(lat)
    g = lat["general_control_deficit"].to_numpy()
    slow = lat["slowness"].to_numpy()
    imp = lat["motor_impulsivity"].to_numpy()
    stop = lat["stop_deficit"].to_numpy()
    att = lat["attention_lapse"].to_numpy()
    var = lat["temporal_variability"].to_numpy()
    conflict = lat["conflict_susceptibility"].to_numpy()

    go_rt = clip(445 + 64 * slow - 30 * imp + 18 * g + rng.normal(0, 45 * noise, n), 220, 950)
    ssrt = clip(245 + 52 * stop + 18 * att + 10 * g + rng.normal(0, 28 * noise, n), 120, 520)
    p_resp_latent = expit(0.00 + 0.80 * stop + 0.42 * imp - 0.20 * slow + 0.18 * att + rng.normal(0, 0.22 * noise, n))
    stop_fail = rng.binomial(52, clip(p_resp_latent, 0.001, 0.999))
    p_resp = 100 * stop_fail / 52
    p_go_om = expit(-4.10 + 0.80 * att + 0.42 * slow + 0.25 * var + rng.normal(0, 0.20 * noise, n))
    p_choice = expit(-3.90 + 0.45 * imp + 0.40 * conflict + 0.28 * att + rng.normal(0, 0.20 * noise, n))
    go_om = rng.binomial(156, clip(p_go_om, 0.001, 0.999))
    go_choice = rng.binomial(156 - go_om, clip(p_choice, 0.001, 0.999))
    tracking_error = clip(np.abs(p_resp - 50) + np.abs(rng.normal(0, 2.5 * noise, n)), 0, 50)

    return pd.DataFrame(
        {
            "sst_p_respond_signal": p_resp,
            "sst_ssrt": ssrt,
            "sst_go_rt": go_rt,
            "sst_go_omission_rate": 100 * go_om / 156,
            "sst_go_choice_error_rate": 100 * go_choice / 156,
            "sst_tracking_error": tracking_error,
        }
    )


def simulate_flanker(lat: pd.DataFrame, rng: np.random.Generator, noise: float) -> pd.DataFrame:
    n = len(lat)
    g = lat["general_control_deficit"].to_numpy()
    slow = lat["slowness"].to_numpy()
    imp = lat["motor_impulsivity"].to_numpy()
    att = lat["attention_lapse"].to_numpy()
    var = lat["temporal_variability"].to_numpy()
    conflict = lat["conflict_susceptibility"].to_numpy()

    rt_cong = clip(430 + 58 * slow - 24 * imp + 12 * g + rng.normal(0, 36 * noise, n), 220, 900)
    effect = clip(78 + 48 * conflict + 18 * att + 10 * g + rng.normal(0, 28 * noise, n), -20, 270)
    interference = clip(effect + rng.normal(0, 12 * noise, n), -25, 290)
    mean_rt = clip(rt_cong + effect / 3 + rng.normal(0, 20 * noise, n), 230, 950)
    cv_rt = clip(17 + 6.0 * var + 2.5 * att + rng.normal(0, 3.5 * noise, n), 4, 70)

    p_err_inc = expit(-3.05 + 0.72 * conflict + 0.34 * imp + 0.22 * att + rng.normal(0, 0.20 * noise, n))
    p_err_cong = expit(-4.65 + 0.30 * imp + 0.20 * att + rng.normal(0, 0.15 * noise, n))
    p_err_neu = expit(-4.20 + 0.25 * conflict + 0.25 * att + rng.normal(0, 0.15 * noise, n))
    err_inc = rng.binomial(48, clip(p_err_inc, 0.001, 0.999))
    err_cong = rng.binomial(48, clip(p_err_cong, 0.001, 0.999))
    err_neu = rng.binomial(48, clip(p_err_neu, 0.001, 0.999))
    p_omiss = expit(-4.60 + 0.65 * att + 0.25 * slow + rng.normal(0, 0.15 * noise, n))
    omiss = rng.binomial(144, clip(p_omiss, 0.001, 0.999))
    accuracy = 100 * (144 - err_inc - err_cong - err_neu - omiss) / 144

    return pd.DataFrame(
        {
            "flanker_effect": effect,
            "flanker_interference": interference,
            "flanker_incongruent_error": 100 * err_inc / 48,
            "flanker_accuracy": clip(accuracy, 0, 100),
            "flanker_mean_rt": mean_rt,
            "flanker_cv_rt": cv_rt,
        }
    )


def simulate_digit_span(lat: pd.DataFrame, rng: np.random.Generator, noise: float) -> pd.DataFrame:
    n = len(lat)
    slow = lat["slowness"].to_numpy()
    imp = lat["motor_impulsivity"].to_numpy()
    att = lat["attention_lapse"].to_numpy()
    wm = lat["verbal_wm_deficit"].to_numpy()
    var = lat["temporal_variability"].to_numpy()

    forward_cont = 6.2 - 0.72 * wm - 0.28 * att + rng.normal(0, 0.55 * noise, n)
    backward_cont = forward_cont - 1.15 - 0.42 * wm - 0.18 * att + rng.normal(0, 0.48 * noise, n)
    forward = bounded_int(forward_cont, 2, 9)
    backward = bounded_int(backward_cont, 2, 8)
    backward = np.minimum(backward, np.maximum(2, forward))
    span_sum = forward + backward
    backward_cost = forward - backward

    accuracy = clip(64 + 4.8 * span_sum - 7.5 * wm - 4.0 * att + rng.normal(0, 7.0 * noise, n), 0, 100)
    response_time = clip(5200 + 750 * slow + 450 * wm + 320 * var - 250 * imp + rng.normal(0, 950 * noise, n), 1500, 16000)

    return pd.DataFrame(
        {
            "ds_forward_span": forward,
            "ds_backward_span": backward,
            "ds_span_sum": span_sum,
            "ds_backward_cost": backward_cost,
            "ds_accuracy": accuracy,
            "ds_response_time": response_time,
        }
    )


def simulate_bacnb(n: int, seed: int, noise: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    lat = sample_latents(n, rng, noise)
    metrics = pd.concat(
        [
            simulate_sart(lat, rng, noise),
            simulate_sst(lat, rng, noise),
            simulate_flanker(lat, rng, noise),
            simulate_digit_span(lat, rng, noise),
        ],
        axis=1,
    )
    metrics.insert(0, "agent_id", np.arange(n))
    return lat, metrics


def fit_deficit_transformer(reference_metrics: pd.DataFrame, clip_limit: float = Z_CLIP) -> dict[str, Any]:
    """Fits a reusable reference for oriented functional z-scores.

    The fit is performed once on the Monte Carlo sample and can then be applied
    to real individuals without recalculating median/MAD from a single person.
    """

    features = {}
    for spec in FEATURES:
        values = reference_metrics[spec.col].to_numpy(dtype=float)
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        mean = float(np.nanmean(values))
        sd = float(np.nanstd(values))
        if np.isfinite(mad) and mad >= 1e-9:
            method = "mad"
        else:
            method = "sd"
        features[spec.col] = {
            "col": spec.col,
            "median": median,
            "mad": mad,
            "mean": mean,
            "sd": sd,
            "higher_worse": bool(spec.higher_worse),
            "clip": float(clip_limit),
            "method": method,
        }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "orientation": "higher_z_deficit_equals_worse",
        "mad_constant": 0.67448975,
        "features": features,
    }


def apply_deficit_transformer(metrics: pd.DataFrame, transformer: dict[str, Any]) -> pd.DataFrame:
    out = {}
    features = transformer["features"]
    for spec in FEATURES:
        ref = features[spec.col]
        values = metrics[spec.col].to_numpy(dtype=float)
        if ref.get("method") == "mad":
            z = 0.67448975 * (values - float(ref["median"])) / (float(ref["mad"]) + 1e-9)
        else:
            z = (values - float(ref["mean"])) / (float(ref["sd"]) + 1e-9)
        if not bool(ref["higher_worse"]):
            z = -z
        out[spec.col] = clip(z, -float(ref["clip"]), float(ref["clip"]))
    return pd.DataFrame(out, index=metrics.index)


def build_deficit_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the previous workflow."""

    return apply_deficit_transformer(metrics, fit_deficit_transformer(metrics))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def silhouette_from_distance(distance: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    scores = []
    for i in range(len(labels)):
        own = labels == labels[i]
        if own.sum() <= 1:
            a = 0.0
        else:
            a = distance[i, own].sum() / (own.sum() - 1)
        b = min(distance[i, labels == lab].mean() for lab in unique if lab != labels[i])
        denom = max(a, b)
        scores.append(0.0 if denom <= 1e-12 else (b - a) / denom)
    return float(np.mean(scores))


def cluster_variables(x_def: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    corr = x_def.corr().fillna(0).to_numpy()
    corr = np.clip(corr, -0.999, 0.999)
    distance = 1 - corr
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    z_link = linkage(condensed, method="average")

    rows = []
    max_k = min(8, len(FEATURES) - 1)
    for k in range(2, max_k + 1):
        labs = fcluster(z_link, t=k, criterion="maxclust")
        sil = silhouette_from_distance(distance, labs)
        min_size = pd.Series(labs).value_counts(normalize=True).min()
        rows.append({"k": k, "silhouette_distancia_correlacao": sil, "menor_cluster_variaveis": min_size})
    criteria = pd.DataFrame(rows)
    valid = criteria[criteria["menor_cluster_variaveis"] >= 0.08]
    best_row = valid.sort_values("silhouette_distancia_correlacao", ascending=False).iloc[0] if len(valid) else criteria.sort_values("silhouette_distancia_correlacao", ascending=False).iloc[0]
    if int(best_row["k"]) < 4:
        detailed = valid[(valid["k"] >= 4) & (valid["silhouette_distancia_correlacao"] >= 0.65 * float(best_row["silhouette_distancia_correlacao"]))]
        if len(detailed):
            best_row = detailed.sort_values("silhouette_distancia_correlacao", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    labels = fcluster(z_link, t=best_k, criterion="maxclust")

    info = []
    for spec, lab in zip(FEATURES, labels):
        info.append(
            {
                "variavel": spec.col,
                "rotulo": spec.label,
                "rotulo_curto": spec.short,
                "teste": spec.test,
                "constructo_emergente_id": int(lab),
            }
        )
    var_info = pd.DataFrame(info)
    construct_names = name_emergent_constructs(var_info)
    var_info["constructo_emergente"] = var_info["constructo_emergente_id"].map(construct_names)
    return var_info, z_link, distance, criteria


def name_emergent_constructs(var_info: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for cid, group in var_info.groupby("constructo_emergente_id"):
        vars_ = set(group["variavel"])
        tests = set(group["teste"])
        if vars_ == {"sst_tracking_error"}:
            name = "Adaptive SSD monitoring"
        elif vars_ == {"ds_backward_cost"}:
            name = "Backward manipulation cost"
        elif len(tests) == 1 and "DS" in tests and {"ds_forward_span", "ds_backward_span", "ds_span_sum"} & vars_:
            name = "Verbal span capacity"
        elif len(tests) == 1 and "DS" in tests:
            name = "Verbal working memory"
        elif len(tests) == 1 and "SST" in tests and {"sst_p_respond_signal", "sst_ssrt"} & vars_:
            name = "Motor cancellation/stop latency"
        elif len(tests) == 1 and "SST" in tests:
            name = "Adaptive SST process"
        elif len(tests) == 1 and "FLANKER" in tests:
            name = "Response conflict"
        elif len(tests) == 1 and "SART" in tests:
            name = "Automation and sustained attention"
        elif any("cv" in v or "omission" in v or "fatigue" in v for v in vars_):
            name = "Cross-test attentional stability"
        elif any("rt" in v or "response_time" in v for v in vars_):
            name = "Speed/temporal strategy"
        elif any("commission" in v or "p_respond" in v or "ssrt" in v or "artifact" in v for v in vars_):
            name = "Prepotent response control"
        elif any("flanker" in v for v in vars_):
            name = "Interference/conflict"
        else:
            name = "Mixed emergent construct"
        names[int(cid)] = f"C{int(cid)} - {name}"
    return names


def compute_emergent_scores(x_def: pd.DataFrame, var_info: pd.DataFrame) -> pd.DataFrame:
    scores = {}
    for cid, group in var_info.groupby("constructo_emergente_id"):
        cols = group["variavel"].tolist()
        name = group["constructo_emergente"].iloc[0]
        scores[name] = x_def[cols].mean(axis=1)
    return pd.DataFrame(scores)


def compute_miyake_scores(x_def: pd.DataFrame) -> pd.DataFrame:
    values = {}
    for key, attr in [
        ("Inhibitory Control", "miyake_inhibition"),
        ("Working Memory", "miyake_working_memory"),
        ("Cognitive Flexibility", "miyake_flexibility"),
    ]:
        num = np.zeros(len(x_def))
        den = 0.0
        for spec in FEATURES:
            w = getattr(spec, attr)
            if w > 0:
                num += w * x_def[spec.col].to_numpy()
                den += w
        values[key] = num / max(den, 1e-9)
    return pd.DataFrame(values)


def pca_scores(x: np.ndarray, n_components: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x0, full_matrices=False)
    explained = (s**2) / np.sum(s**2)
    n_components = min(n_components, x.shape[1])
    scores = u[:, :n_components] * s[:n_components]
    return scores, explained[:n_components], vt[:n_components]


def kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=float)
    idx = rng.integers(0, n)
    centers[0] = x[idx]
    dist_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = dist_sq / (dist_sq.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centers[i] = x[idx]
        dist_sq = np.minimum(dist_sq, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def assign_centers(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist_sq = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = dist_sq.argmin(axis=1)
    min_dist = dist_sq[np.arange(x.shape[0]), labels]
    return labels, min_dist


def run_kmeans(x: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 8, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray, float]:
    best_labels = None
    best_centers = None
    best_inertia = float("inf")
    for _ in range(n_init):
        centers = kmeans_pp_init(x, k, rng)
        labels = np.zeros(x.shape[0], dtype=int)
        for _it in range(max_iter):
            new_labels, min_dist = assign_centers(x, centers)
            if np.array_equal(new_labels, labels) and _it > 0:
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.sum() == 0:
                    centers[j] = x[rng.integers(0, x.shape[0])]
                else:
                    centers[j] = x[mask].mean(axis=0)
        labels, min_dist = assign_centers(x, centers)
        inertia = float(min_dist.sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers, best_inertia


def centroid_silhouette(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    dist = np.sqrt(((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    own = dist[np.arange(x.shape[0]), labels]
    alt = dist.copy()
    alt[np.arange(x.shape[0]), labels] = np.inf
    other = np.min(alt, axis=1)
    score = (other - own) / (np.maximum(other, own) + 1e-12)
    return float(np.mean(score))


def choose_agent_k(scores: np.ndarray, seed: int, max_k: int, sample_k: int) -> tuple[int, pd.DataFrame]:
    rng = np.random.default_rng(seed + 11)
    n = scores.shape[0]
    idx = rng.choice(n, size=min(sample_k, n), replace=False)
    x = scores[idx]
    rows = []
    for k in range(2, max_k + 1):
        labels, centers, inertia = run_kmeans(x, k, rng, n_init=5, max_iter=60)
        sizes = np.bincount(labels, minlength=k) / len(labels)
        sil = centroid_silhouette(x, labels, centers)
        rows.append(
            {
                "k": k,
                "inertia": inertia,
                "centroid_silhouette": sil,
                "menor_cluster": sizes.min(),
                "maior_cluster": sizes.max(),
            }
        )
    crit = pd.DataFrame(rows)
    crit["inertia_delta_rel"] = crit["inertia"].pct_change().abs().fillna(np.nan)
    valid = crit[crit["menor_cluster"] >= 0.03]
    chosen = valid.sort_values(["centroid_silhouette", "k"], ascending=[False, True]).iloc[0] if len(valid) else crit.sort_values(["centroid_silhouette", "k"], ascending=[False, True]).iloc[0]
    if int(chosen["k"]) < 4:
        # K=2 solutions often summarize everything as a general severity axis.
        # For profile auditing, a more detailed solution is accepted if it
        # retains at least 70% of the best silhouette and non-residual clusters.
        detailed = valid[(valid["k"] >= 4) & (valid["centroid_silhouette"] >= 0.70 * float(chosen["centroid_silhouette"]))]
        if len(detailed):
            chosen = detailed.sort_values(["centroid_silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(chosen["k"]), crit


def reorder_clusters(labels: np.ndarray, x_def: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    tmp = pd.DataFrame({"cluster": labels, "deficit_global": x_def.mean(axis=1)})
    order = tmp.groupby("cluster")["deficit_global"].mean().sort_values().index.tolist()
    mapping = {old: i + 1 for i, old in enumerate(order)}
    new = np.array([mapping[x] for x in labels], dtype=int)
    return new, mapping


def label_agent_clusters(cluster_centroids: pd.DataFrame, construct_scores: pd.DataFrame | None = None) -> dict[int, str]:
    labels = {}
    rt_metrics = {"sart_mean_rt", "sst_go_rt", "flanker_mean_rt", "ds_response_time"}
    for cluster_id, row in cluster_centroids.iterrows():
        if int(cluster_id) in PROFILE_NAMES:
            labels[int(cluster_id)] = f"P{int(cluster_id)} - {PROFILE_NAMES[int(cluster_id)]}"
            continue
        high = row.sort_values(ascending=False)
        top = high.index[:5].tolist()
        mean_deficit = row.mean()
        if mean_deficit < -0.35:
            name = "Flexible regulation"
        elif mean_deficit > 0.95 and sum(v > 0.75 for v in row.to_numpy()) >= 9:
            name = "Global control economy"
        elif any("omission" in c or "cv" in c or "fatigue" in c for c in top[:3]):
            name = "Exploratory variability"
        elif any(c in rt_metrics for c in top[:3]):
            name = "Strategic slowness/elevated time"
        elif any(c.startswith("ds_") for c in top[:3]):
            name = "Vulnerable verbal working memory"
        elif any(c in top[:4] for c in ["sst_p_respond_signal", "sst_ssrt", "sst_tracking_error"]):
            name = "Slow or unstable motor cancellation"
        elif any(c.startswith("flanker_") for c in top[:3]):
            name = "Elevated conflict/interference"
        elif any(c.startswith("sart_") for c in top[:3]) and ("sart_commission_rate" in top or "sart_automation_index" in top):
            name = "Prepotent automation"
        else:
            name = "Mixed functional profile"
        labels[int(cluster_id)] = f"P{int(cluster_id)} - {name}"
    return labels


def aggregate_by_test(x_def: pd.DataFrame) -> pd.DataFrame:
    data = {}
    for test in sorted({spec.test for spec in FEATURES}):
        cols = [spec.col for spec in FEATURES if spec.test == test]
        data[TEST_LABELS[test]] = x_def[cols].mean(axis=1)
    return pd.DataFrame(data)


def symmetric_limits(mat: pd.DataFrame | np.ndarray) -> tuple[float, float]:
    arr = mat.to_numpy(dtype=float) if isinstance(mat, pd.DataFrame) else np.asarray(mat, dtype=float)
    max_abs = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
    lim = max(0.5, math.ceil(max_abs * 10) / 10)
    return -lim, lim


def profile_label(pid: int) -> str:
    return f"{PROFILE_SHORT.get(int(pid), f'P{pid}')} - {PROFILE_NAMES.get(int(pid), str(pid))}"


def profile_from_similarity_col(col: str) -> str:
    for pid, sim_col in PROFILE_COLUMNS.items():
        if col == sim_col:
            return PROFILE_SHORT[pid]
    return col


def pretty_state_label(state: str) -> str:
    mapping = {
        "ESTADO_ASSIMETRICO_DISSOCIADO": "Asymmetric/dissociated",
        "ESTADO_PROTOTIPICO_P1_CONTROLE_PRESERVADO": "Prototypical P1",
        "ESTADO_PROTOTIPICO_P2_LAPSOS_VARIABILIDADE": "Prototypical P2",
        "ESTADO_PROTOTIPICO_P3_AUTOMATIZACAO_PREPOTENTE": "Prototypical P3",
        "ESTADO_PROTOTIPICO_P4_DEFICIT_AMPLO_CONVERGENTE": "Prototypical P4",
        "ESTADO_MISTO_COM_COMPONENTE_P4_SEM_CONVERGENCIA_GLOBAL": "Mixed with P4 without broad gate",
        "ESTADO_SOBREPOSTO_MULTIPERFIL": "Overlapping multiprofile",
        "ESTADO_EXTREMO_FORA_DOS_PROTOTIPOS": "Outside prototypes",
        "OUTROS_ESTADOS_MISTOS_RAROS": "Other rare mixed states",
    }
    if state in mapping:
        return mapping[state]
    if state.startswith("ESTADO_MISTO_"):
        return state.replace("ESTADO_MISTO_", "Mixed ").replace("_", "-")
    return state.replace("ESTADO_", "").replace("_", " ").title()


def regularized_inverse_cov(x: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] <= 2:
        cov = np.eye(x.shape[1])
    else:
        cov = np.cov(x, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = cov + np.eye(cov.shape[0]) * ridge
    try:
        cond = np.linalg.cond(cov)
        if not np.isfinite(cond) or cond > 1e8:
            return np.linalg.pinv(cov)
        return np.linalg.inv(cov)
    except Exception:
        try:
            return np.linalg.pinv(cov)
        except Exception:
            diag = np.diag(np.diag(cov))
            diag = diag + np.eye(diag.shape[0]) * ridge
            return np.linalg.pinv(diag)


def calcular_prototipos_perfis(
    x_def: pd.DataFrame,
    clusters: pd.Series,
    cluster_labels: dict[int, str],
    emergent_scores: pd.DataFrame | None = None,
    test_scores: pd.DataFrame | None = None,
) -> dict[str, Any]:
    metric_centroids = x_def.copy()
    metric_centroids["perfil_id"] = clusters.to_numpy()
    metric_centroids = metric_centroids.groupby("perfil_id").mean()
    metric_centroids.index.name = "perfil_id"

    construct_centroids = pd.DataFrame()
    if emergent_scores is not None:
        construct_centroids = emergent_scores.copy()
        construct_centroids["perfil_id"] = clusters.to_numpy()
        construct_centroids = construct_centroids.groupby("perfil_id").mean()
        construct_centroids.index.name = "perfil_id"

    test_centroids = pd.DataFrame()
    if test_scores is not None:
        test_centroids = test_scores.copy()
        test_centroids["perfil_id"] = clusters.to_numpy()
        test_centroids = test_centroids.groupby("perfil_id").mean()
        test_centroids.index.name = "perfil_id"

    inverse_covariances: dict[int, np.ndarray] = {}
    distance_reference: dict[int, dict[str, float]] = {}
    for pid in metric_centroids.index:
        pid_int = int(pid)
        subset = x_def.loc[clusters.to_numpy() == pid_int, metric_centroids.columns].to_numpy(dtype=float)
        inv = regularized_inverse_cov(subset)
        inverse_covariances[pid_int] = inv
        delta = subset - metric_centroids.loc[pid].to_numpy(dtype=float)
        dist = np.sqrt(np.maximum(np.sum((delta @ inv) * delta, axis=1), 0.0))
        distance_reference[pid_int] = {
            "p50": float(np.nanpercentile(dist, 50)),
            "p95": float(np.nanpercentile(dist, 95)),
            "p99": float(np.nanpercentile(dist, 99)),
            "max": float(np.nanmax(dist)),
            "n": int(subset.shape[0]),
        }

    return {
        "profile_names": {int(k): cluster_labels.get(int(k), profile_label(int(k))) for k in metric_centroids.index},
        "metric_centroids": metric_centroids,
        "construct_centroids": construct_centroids,
        "test_centroids": test_centroids,
        "inverse_covariances": inverse_covariances,
        "distance_reference": distance_reference,
    }


def calcular_distancias_perfis(x_def: pd.DataFrame, prototipos: dict[str, Any], metodo: str = "mahalanobis", agent_ids: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    cols = prototipos["metric_centroids"].columns.tolist()
    x = x_def[cols].to_numpy(dtype=float)
    out = {"agent_id": np.arange(len(x_def)) if agent_ids is None else np.asarray(agent_ids)}
    method_used = []
    for pid, centroid_row in prototipos["metric_centroids"].iterrows():
        pid_int = int(pid)
        centroid = centroid_row.to_numpy(dtype=float)
        delta = x - centroid
        used = metodo
        if metodo == "mahalanobis":
            try:
                inv = prototipos["inverse_covariances"][pid_int]
                dist_sq = np.sum((delta @ inv) * delta, axis=1)
                dist = np.sqrt(np.maximum(dist_sq, 0.0))
                if not np.isfinite(dist).all():
                    raise FloatingPointError("non-finite Mahalanobis distance")
            except Exception:
                used = "euclidean_fallback"
                dist = np.sqrt(np.sum(delta**2, axis=1))
        else:
            used = "euclidean"
            dist = np.sqrt(np.sum(delta**2, axis=1))
        out[DISTANCE_COLUMNS[pid_int]] = dist
        method_used.append(used)
    out["metodo_distancia"] = metodo if len(set(method_used)) == 1 and method_used[0] == metodo else "+".join(sorted(set(method_used)))
    return pd.DataFrame(out)


def calcular_similaridade_funcional(distancias: pd.DataFrame, tau: float = 1.0) -> pd.DataFrame:
    tau = max(float(tau), 1e-6)
    dcols = [DISTANCE_COLUMNS[pid] for pid in sorted(DISTANCE_COLUMNS)]
    distances = distancias[dcols].to_numpy(dtype=float)
    scores = -(distances**2) / (2 * tau**2)
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
    out = pd.DataFrame({"agent_id": distancias["agent_id"].to_numpy()})
    for i, pid in enumerate(sorted(PROFILE_COLUMNS)):
        out[PROFILE_COLUMNS[pid]] = weights[:, i]
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    order = np.argsort(-weights, axis=1)
    out["perfil_predominante"] = [profile_from_similarity_col(sim_cols[i]) for i in order[:, 0]]
    out["perfil_secundario"] = [profile_from_similarity_col(sim_cols[i]) for i in order[:, 1]]
    out["similaridade_top1"] = weights[np.arange(len(weights)), order[:, 0]]
    out["similaridade_top2"] = weights[np.arange(len(weights)), order[:, 1]]
    out["margem_top1_top2"] = out["similaridade_top1"] - out["similaridade_top2"]
    return out


def calcular_vetor_estado_unitario(similaridades: pd.DataFrame) -> pd.DataFrame:
    """Converts P1-P4 weights into a unit compositional vector.

    Similarity columns are softmax-normalized weights and sum to 1 for
    each agent. The unit vector is the explicit clinical and geometric form of
    the same state: v = [P1, P2, P3, P4], with sum equal to 1.00.
    """
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    labels = [PROFILE_SHORT[pid] for pid in sorted(PROFILE_COLUMNS)]
    w = similaridades[sim_cols].to_numpy(dtype=float)
    row_sum = w.sum(axis=1, keepdims=True)
    w = w / (row_sum + 1e-12)

    out = pd.DataFrame({"agent_id": similaridades["agent_id"].to_numpy()})
    for i, label in enumerate(labels):
        out[label] = w[:, i]
    out["vetor_estado_unitario"] = [
        "[" + "; ".join(f"{row[i]:.4f}" for i, _label in enumerate(labels)) + "]"
        for row in w
    ]
    out["perfil_predominante"] = similaridades["perfil_predominante"].to_numpy()
    out["perfil_secundario"] = similaridades["perfil_secundario"].to_numpy()
    out["margem_top1_top2"] = similaridades["margem_top1_top2"].to_numpy(dtype=float)
    return out


def calcular_indice_sobreposicao(similaridades: pd.DataFrame) -> pd.DataFrame:
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    w = similaridades[sim_cols].to_numpy(dtype=float)
    sorted_w = np.sort(w, axis=1)[:, ::-1]
    entropy = -np.sum(w * np.log(w + 1e-12), axis=1)
    entropy_norm = entropy / math.log(len(sim_cols))
    ratio = sorted_w[:, 0] / (sorted_w[:, 1] + 1e-12)
    top12 = sorted_w[:, 0] + sorted_w[:, 1]
    classificacao = np.where(
        (sorted_w[:, 0] >= 0.70) & ((sorted_w[:, 0] - sorted_w[:, 1]) >= 0.35),
        "perfil_relativamente_puro",
        np.where(entropy_norm >= 0.75, "perfil_altamente_sobreposto", "perfil_misto"),
    )
    return pd.DataFrame(
        {
            "agent_id": similaridades["agent_id"].to_numpy(),
            "entropia_similaridade": entropy,
            "entropia_normalizada": entropy_norm,
            "margem_top1_top2": sorted_w[:, 0] - sorted_w[:, 1],
            "razao_top1_top2": ratio,
            "soma_top1_top2": top12,
            "classificacao_sobreposicao": classificacao,
        }
    )


def verificar_gate_deficit_amplo(test_scores: pd.DataFrame, construct_scores: pd.DataFrame, x_def: pd.DataFrame) -> pd.Series:
    deficit_global = x_def.mean(axis=1)
    n_testes = (test_scores >= 0.75).sum(axis=1)
    n_constructos_075 = (construct_scores >= 0.75).sum(axis=1)
    n_constructos_100 = (construct_scores >= 1.00).sum(axis=1)
    return (deficit_global >= 0.75) & (n_testes >= 3) & ((n_constructos_075 >= 4) | (n_constructos_100 >= 3))


def detectar_dissociacao_funcional(x_def: pd.DataFrame, construct_scores: pd.DataFrame, test_scores: pd.DataFrame) -> pd.DataFrame:
    deficit_global = x_def.mean(axis=1)
    n_constructos = (construct_scores >= 1.25).sum(axis=1)
    n_testes = (test_scores >= 0.75).sum(axis=1)
    n_extremas = (x_def >= 2.5).sum(axis=1)
    dominio_preservado = (test_scores <= 0.0).any(axis=1) | (construct_scores <= 0.0).any(axis=1)

    sart_auto = (x_def.get("sart_commission_rate", 0) >= 2.0) | (x_def.get("sart_automation_index", 0) >= 2.0) | (x_def.get("sart_artifact_rate", 0) >= 2.0)
    span_vulneravel = (x_def.get("ds_backward_span", 0) >= 1.25) | (x_def.get("ds_span_sum", 0) >= 1.25) | (x_def.get("ds_backward_cost", 0) >= 1.25)
    conflito = (x_def.get("flanker_effect", 0) >= 1.5) | (x_def.get("flanker_incongruent_error", 0) >= 1.5)
    cancelamento = (x_def.get("sst_p_respond_signal", 0) >= 1.5) | (x_def.get("sst_ssrt", 0) >= 1.5)
    velocidade = (x_def.get("sart_mean_rt", 0) >= 1.5) | (x_def.get("sst_go_rt", 0) >= 1.5) | (x_def.get("flanker_mean_rt", 0) >= 1.5) | (x_def.get("ds_response_time", 0) >= 1.5)

    diss = (deficit_global < 0.75) & (n_constructos <= 2) & (n_testes <= 2) & (n_extremas >= 1) & dominio_preservado
    subtipo = np.full(len(x_def), "SEM_DISSOCIACAO", dtype=object)
    subtipo = np.where(diss & sart_auto, "DISSOCIACAO_AUTOMATIZACAO_SART", subtipo)
    subtipo = np.where(diss & span_vulneravel & (subtipo == "SEM_DISSOCIACAO"), "DISSOCIACAO_SPAN_VERBAL", subtipo)
    subtipo = np.where(diss & conflito & (subtipo == "SEM_DISSOCIACAO"), "DISSOCIACAO_CONFLITO_FLANKER", subtipo)
    subtipo = np.where(diss & cancelamento & (subtipo == "SEM_DISSOCIACAO"), "DISSOCIACAO_CANCELAMENTO_SST", subtipo)
    subtipo = np.where(diss & velocidade & (subtipo == "SEM_DISSOCIACAO"), "DISSOCIACAO_VELOCIDADE", subtipo)
    subtipo = np.where(diss & (subtipo == "SEM_DISSOCIACAO"), "DISSOCIACAO_MISTA_FOCAL", subtipo)

    return pd.DataFrame(
        {
            "dissociacao_funcional": diss.to_numpy(dtype=bool),
            "subtipo_dissociacao": subtipo,
            "deficit_global_medio": deficit_global.to_numpy(dtype=float),
            "n_testes_alterados": n_testes.to_numpy(dtype=int),
            "n_constructos_alterados": (construct_scores >= 0.75).sum(axis=1).to_numpy(dtype=int),
            "n_metricas_extremas": n_extremas.to_numpy(dtype=int),
        },
        index=x_def.index,
    )


def calibrar_espaco_estados(
    similaridades: pd.DataFrame,
    distancias: pd.DataFrame,
    clusters: pd.Series,
    x_def: pd.DataFrame,
    construct_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
) -> dict[str, Any]:
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    top = similaridades["similaridade_top1"].to_numpy(dtype=float)
    margem = similaridades["margem_top1_top2"].to_numpy(dtype=float)
    sobre = calcular_indice_sobreposicao(similaridades)
    deficit_global = x_def.mean(axis=1).to_numpy(dtype=float)
    n_testes = (test_scores >= 0.75).sum(axis=1).to_numpy(dtype=int)
    n_constructos = (construct_scores >= 0.75).sum(axis=1).to_numpy(dtype=int)
    n_extremas = (x_def >= 2.5).sum(axis=1).to_numpy(dtype=int)
    per_profile = {}
    for pid in sorted(PROFILE_NAMES):
        mask = clusters.to_numpy() == pid
        dcol = DISTANCE_COLUMNS[pid]
        vals = distancias.loc[mask, dcol].to_numpy(dtype=float)
        per_profile[str(pid)] = {
            "dist_p95": float(np.nanpercentile(vals, 95)),
            "dist_p99": float(np.nanpercentile(vals, 99)),
            "top1_p25": float(np.nanpercentile(top[mask], 25)),
            "top1_p50": float(np.nanpercentile(top[mask], 50)),
            "margem_p25": float(np.nanpercentile(margem[mask], 25)),
            "n": int(mask.sum()),
        }
    return {
        "similarity_columns": sim_cols,
        "distance_columns": [DISTANCE_COLUMNS[pid] for pid in sorted(DISTANCE_COLUMNS)],
        "profile_reference": per_profile,
        "global": {
            "top1_p25": float(np.nanpercentile(top, 25)),
            "top1_p50": float(np.nanpercentile(top, 50)),
            "top1_p75": float(np.nanpercentile(top, 75)),
            "margem_p25": float(np.nanpercentile(margem, 25)),
            "entropia_norm_p75": float(np.nanpercentile(sobre["entropia_normalizada"], 75)),
            "entropia_norm_p90": float(np.nanpercentile(sobre["entropia_normalizada"], 90)),
            "deficit_global_p75": float(np.nanpercentile(deficit_global, 75)),
            "deficit_global_p90": float(np.nanpercentile(deficit_global, 90)),
            "n_testes_alterados_p75": float(np.nanpercentile(n_testes, 75)),
            "n_constructos_alterados_p75": float(np.nanpercentile(n_constructos, 75)),
            "n_metricas_extremas_p75": float(np.nanpercentile(n_extremas, 75)),
        },
    }


def categorizar_estado_funcional(
    similaridades: pd.DataFrame,
    distancias: pd.DataFrame,
    x_def: pd.DataFrame,
    construct_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    referencia_estado: dict[str, Any] | None = None,
) -> pd.DataFrame:
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    w = similaridades[sim_cols].to_numpy(dtype=float)
    order = np.argsort(-w, axis=1)
    sorted_w = np.take_along_axis(w, order, axis=1)
    pids = np.array(sorted(PROFILE_COLUMNS))
    top_ids = pids[order]
    entropy = -np.sum(w * np.log(w + 1e-12), axis=1)
    entropy_norm = entropy / math.log(len(sim_cols))
    dcols = [DISTANCE_COLUMNS[pid] for pid in sorted(DISTANCE_COLUMNS)]
    d = distancias[dcols].to_numpy(dtype=float)
    min_dist = np.nanmin(d, axis=1)
    deficit_global = x_def.mean(axis=1)
    n_testes = (test_scores >= 0.75).sum(axis=1)
    n_constructos = (construct_scores >= 0.75).sum(axis=1)
    n_extremas = (x_def >= 2.5).sum(axis=1)
    diss = detectar_dissociacao_funcional(x_def, construct_scores, test_scores)
    broad_gate = verificar_gate_deficit_amplo(test_scores, construct_scores, x_def)

    ood = np.zeros(len(similaridades), dtype=bool)
    if referencia_estado:
        profile_ref = referencia_estado.get("profile_reference", {})
        for i, pid in enumerate(top_ids[:, 0]):
            p99 = profile_ref.get(str(int(pid)), {}).get("dist_p99", float("inf"))
            dcol_idx = sorted(DISTANCE_COLUMNS).index(int(pid))
            ood[i] = d[i, dcol_idx] > p99
    ood = ood | (sorted_w[:, 0] < 0.35)

    estado = []
    subtipo = []
    interpretacao = []
    for i in range(len(similaridades)):
        top1_pid = int(top_ids[i, 0])
        top2_pid = int(top_ids[i, 1])
        top3_pid = int(top_ids[i, 2])
        top1 = float(sorted_w[i, 0])
        top2 = float(sorted_w[i, 1])
        top3 = float(sorted_w[i, 2])
        margem = top1 - top2
        dissociado = bool(diss.iloc[i]["dissociacao_funcional"])
        gate_p4 = bool(broad_gate.iloc[i])

        if dissociado:
            est = "ESTADO_ASSIMETRICO_DISSOCIADO"
            sub = str(diss.iloc[i]["subtipo_dissociacao"])
            txt = "Asymmetric/dissociated functional state, with focal or selective alteration without a broad executive-deficit pattern."
        elif top1 >= 0.70 and margem >= 0.35:
            if top1_pid == 4 and not gate_p4:
                est = "ESTADO_MISTO_COM_COMPONENTE_P4_SEM_CONVERGENCIA_GLOBAL"
                sub = f"SEM_GATE_P4_{PROFILE_SHORT[top2_pid]}"
                txt = "Greater approximation to P4 without a broad convergent gate; interpret as a mixed component, not as fully convergent global control economy."
            else:
                if top1_pid == 1:
                    est = "ESTADO_PROTOTIPICO_P1_CONTROLE_PRESERVADO"
                    txt = "Functional state close to the preserved/efficient-control prototype."
                elif top1_pid == 2:
                    est = "ESTADO_PROTOTIPICO_P2_LAPSOS_VARIABILIDADE"
                    txt = "Functional state close to the attentional-lapse and variability prototype."
                elif top1_pid == 3:
                    est = "ESTADO_PROTOTIPICO_P3_AUTOMATIZACAO_PREPOTENTE"
                    txt = "Functional state close to the rigid/prepotent automation prototype."
                else:
                    est = "ESTADO_PROTOTIPICO_P4_DEFICIT_AMPLO_CONVERGENTE"
                    txt = "Functional state close to the broad/mixed executive-deficit prototype, with convergent impairment across multiple domains."
                sub = f"PROTOTIPICO_{PROFILE_SHORT[top1_pid]}"
        elif ood[i]:
            est = "ESTADO_EXTREMO_FORA_DOS_PROTOTIPOS"
            sub = "FORA_DA_REGIAO_BEM_REPRESENTADA"
            txt = "Extreme functional state or poorly represented by current ideal prototypes; requires manual inspection."
        elif top1 < 0.55 and top2 >= 0.20 and top3 >= 0.15 and entropy_norm[i] >= 0.75:
            est = "ESTADO_SOBREPOSTO_MULTIPERFIL"
            sub = f"SOBREPOSICAO_{PROFILE_SHORT[top1_pid]}_{PROFILE_SHORT[top2_pid]}_{PROFILE_SHORT[top3_pid]}"
            txt = "Highly overlapping functional state, without clear dominance by a single prototype."
        elif top1 + top2 >= 0.70 and top2 >= 0.25:
            est = f"ESTADO_MISTO_{PROFILE_SHORT[top1_pid]}_{PROFILE_SHORT[top2_pid]}"
            sub = "TRANSICAO_ENTRE_PROTOTIPOS"
            txt = "Mixed functional state located in the transition region between two ideal prototypes."
        else:
            est = "ESTADO_SOBREPOSTO_MULTIPERFIL"
            sub = "SOBREPOSICAO_DIFUSA"
            txt = "Overlapping or diffuse functional state; interpret using dimensional weights and altered constructs."

        estado.append(est)
        subtipo.append(sub)
        interpretacao.append(txt)

    return pd.DataFrame(
        {
            "agent_id": similaridades["agent_id"].to_numpy(),
            "estado_funcional": estado,
            "subtipo_estado": subtipo,
            "perfil_predominante": [PROFILE_SHORT[int(x)] for x in top_ids[:, 0]],
            "perfil_secundario": [PROFILE_SHORT[int(x)] for x in top_ids[:, 1]],
            "top1": sorted_w[:, 0],
            "top2": sorted_w[:, 1],
            "top3": sorted_w[:, 2],
            "margem_top1_top2": sorted_w[:, 0] - sorted_w[:, 1],
            "entropia_similaridade": entropy,
            "entropia_normalizada": entropy_norm,
            "distancia_minima_prototipo": min_dist,
            "out_of_distribution": ood,
            "deficit_global_medio": deficit_global.to_numpy(dtype=float),
            "n_testes_alterados": n_testes.to_numpy(dtype=int),
            "n_constructos_alterados": n_constructos.to_numpy(dtype=int),
            "n_metricas_extremas": n_extremas.to_numpy(dtype=int),
            "dissociacao_funcional": diss["dissociacao_funcional"].to_numpy(dtype=bool),
            "subtipo_dissociacao": diss["subtipo_dissociacao"].to_numpy(),
            "interpretacao_curta_estado": interpretacao,
        }
    )


def criar_casos_sinteticos_validacao(prototipos: dict[str, Any] | None = None) -> pd.DataFrame:
    """Creates only ideal P1-P4 cases to audit the state space.

    These cases do not represent diagnoses or particular clinical profiles. They are
    ideal anchor points used to test whether the state space recognizes the four
    prototypical regions derived from the simulation itself.
    """
    base = {spec.col: 0.0 for spec in FEATURES}
    if prototipos is not None:
        cent = prototipos["metric_centroids"]
        proto_base = {int(pid): row.to_dict() for pid, row in cent.iterrows()}
    else:
        proto_base = {}

    case_defs = [
        (1, "Ideal P1 case - Preserved control"),
        (2, "Ideal P2 case - Lapses and variability"),
        (3, "Ideal P3 case - Prepotent automation"),
        (4, "Ideal P4 case - Global control economy"),
    ]
    cases = []
    for pid, case_id in case_defs:
        row = base.copy()
        row.update(proto_base.get(pid, {}))
        row["case_id"] = case_id
        cases.append(row)
    df = pd.DataFrame(cases)
    return df[["case_id"] + [spec.col for spec in FEATURES]]

def interpretar_perfil_individual(
    x_def_individual: pd.Series,
    similaridade_row: pd.Series,
    estado_row: pd.Series,
    construct_scores: pd.Series,
    test_scores: pd.Series,
) -> str:
    top_constructs = construct_scores.sort_values(ascending=False).head(3)
    preserved_constructs = construct_scores.sort_values().head(2)
    altered_tests = test_scores.sort_values(ascending=False).head(2)
    preserved_tests = test_scores.sort_values().head(2)
    return (
        f"Highest functional similarity with {similaridade_row['perfil_predominante']} "
        f"({similaridade_row['similaridade_top1']:.2f}), secondary approximation to "
        f"{similaridade_row['perfil_secundario']} ({similaridade_row['similaridade_top2']:.2f}). "
        f"State: {estado_row['estado_funcional']}. "
        f"Constructs that most shift the case: {', '.join(f'{k}={v:+.2f}' for k, v in top_constructs.items())}. "
        f"Relatively preserved constructs: {', '.join(f'{k}={v:+.2f}' for k, v in preserved_constructs.items())}. "
        f"Most altered tests: {', '.join(f'{k}={v:+.2f}' for k, v in altered_tests.items())}. "
        f"Preserved tests: {', '.join(f'{k}={v:+.2f}' for k, v in preserved_tests.items())}. "
        f"{estado_row['interpretacao_curta_estado']}"
    )


def build_similarity_space(x_def: pd.DataFrame, emergent_scores: pd.DataFrame, space: str) -> pd.DataFrame:
    if space == "constructs":
        return emergent_scores.copy()
    if space == "both":
        return pd.concat(
            [
                x_def.add_prefix("metrica__"),
                emergent_scores.add_prefix("constructo__"),
            ],
            axis=1,
        )
    return x_def.copy()


def compute_case_outputs(
    cases_z: pd.DataFrame,
    distance_prototipos: dict[str, Any],
    var_info: pd.DataFrame,
    tau: float,
    method: str,
    referencia_estado: dict[str, Any],
    similarity_space: str = "metrics",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_cases = cases_z[[spec.col for spec in FEATURES]].copy()
    case_ids = cases_z["case_id"]
    case_constructs = compute_emergent_scores(x_cases, var_info)
    case_tests = aggregate_by_test(x_cases)
    case_similarity_space = build_similarity_space(x_cases, case_constructs, similarity_space)
    case_dist = calcular_distancias_perfis(case_similarity_space, distance_prototipos, metodo=method, agent_ids=case_ids)
    case_sim = calcular_similaridade_funcional(case_dist, tau=tau)
    case_estado = categorizar_estado_funcional(case_sim, case_dist, x_cases, case_constructs, case_tests, referencia_estado)
    case_sim.insert(0, "case_id", case_ids.to_numpy())
    case_estado.insert(0, "case_id", case_ids.to_numpy())
    case_constructs.insert(0, "case_id", case_ids.to_numpy())
    case_tests.insert(0, "case_id", case_ids.to_numpy())
    return case_dist, case_sim, case_estado, case_constructs, case_tests


def fig_similarity_profiles(outdir: Path, similaridades: pd.DataFrame, estado: pd.DataFrame, seed: int) -> None:
    rng = np.random.default_rng(seed + 37)
    idx = rng.choice(len(similaridades), size=min(260, len(similaridades)), replace=False)
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    mat = similaridades.iloc[idx][sim_cols].copy()
    order = np.argsort(-mat.to_numpy().max(axis=1))
    mat = mat.iloc[order]
    mat.columns = [PROFILE_SHORT[pid] for pid in sorted(PROFILE_COLUMNS)]
    fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
    sns.heatmap(mat, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "functional similarity"}, yticklabels=False, ax=ax)
    ax.set_title("BACNB Monte Carlo - Functional Similarity to Prototypes")
    ax.set_xlabel("Ideal functional prototype")
    ax.set_ylabel("Synthetic-agent sample")
    fig.savefig(outdir / "BACNB_Fig11_SimilaridadeFuncionalPerfis.png", bbox_inches="tight")
    plt.close(fig)


def fig_synthetic_decomposition(outdir: Path, case_sim: pd.DataFrame) -> None:
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    plot = case_sim.set_index("case_id")[sim_cols] * 100
    plot.columns = [PROFILE_SHORT[pid] for pid in sorted(PROFILE_COLUMNS)]
    fig, ax = plt.subplots(figsize=(16, 8.5), constrained_layout=True)
    bottom = np.zeros(len(plot))
    colors = sns.color_palette("tab10", len(plot.columns))
    y = np.arange(len(plot))
    for col, color in zip(plot.columns, colors):
        ax.barh(y, plot[col].to_numpy(), left=bottom, label=col, color=color, edgecolor="black", linewidth=0.8)
        bottom += plot[col].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels(plot.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Functional similarity (%)")
    ax.set_ylabel("Synthetic case")
    ax.set_title("BACNB Monte Carlo - Synthetic Case Decomposition")
    ax.legend(title="Prototype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.savefig(outdir / "BACNB_Fig12_DecomposicaoCasosSinteticos.png", bbox_inches="tight")
    plt.close(fig)


def fig_synthetic_constructs(outdir: Path, case_constructs: pd.DataFrame) -> None:
    mat = case_constructs.set_index("case_id")
    vmin, vmax = symmetric_limits(mat)
    fig, ax = plt.subplots(figsize=(18, 8.5), constrained_layout=True)
    sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=vmin, vmax=vmax, annot=True, fmt=".2f", linewidths=0.5, linecolor="white", cbar_kws={"label": "oriented functional z"}, ax=ax)
    ax.set_title("BACNB Monte Carlo - Construct Dissociation in Synthetic Cases")
    ax.set_xlabel("Emergent construct")
    ax.set_ylabel("Synthetic case")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.savefig(outdir / "BACNB_Fig13_DissociacaoConstructosCasosSinteticos.png", bbox_inches="tight")
    plt.close(fig)


def pca_2d_from_matrix(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores, explained, _vt = pca_scores(mat, n_components=2)
    return scores, explained


def fig_state_space_simplex(outdir: Path, similaridades: pd.DataFrame, estado: pd.DataFrame, case_sim: pd.DataFrame | None, seed: int) -> None:
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    rng = np.random.default_rng(seed + 41)
    idx = rng.choice(len(similaridades), size=min(12000, len(similaridades)), replace=False)
    all_sim = similaridades[sim_cols].to_numpy(dtype=float)
    scores, explained = pca_2d_from_matrix(all_sim)
    state_raw = estado["estado_funcional"].copy()
    counts = state_raw.value_counts()
    min_count = max(50, int(0.01 * len(state_raw)))
    state_grouped = state_raw.where(state_raw.map(counts) >= min_count, "OUTROS_ESTADOS_MISTOS_RAROS")
    plot = pd.DataFrame({"PC1": scores[idx, 0], "PC2": scores[idx, 1], "State": state_grouped.iloc[idx].map(pretty_state_label).to_numpy()})
    fig, ax = plt.subplots(figsize=(17, 10), constrained_layout=True)
    sns.scatterplot(data=plot, x="PC1", y="PC2", hue="State", s=16, alpha=0.35, linewidth=0, palette="tab10", ax=ax)
    if case_sim is not None and len(case_sim):
        case_scores = (case_sim[sim_cols].to_numpy(dtype=float) - all_sim.mean(axis=0)) @ np.linalg.svd(all_sim - all_sim.mean(axis=0), full_matrices=False)[2][:2].T
        ax.scatter(case_scores[:, 0], case_scores[:, 1], s=140, c="black", marker="*", label="Synthetic cases", zorder=5)
        offsets = [(0.01, 0.02), (0.01, -0.03), (-0.05, -0.03), (0.02, 0.02), (0.03, -0.03)]
        for j, (x, y) in enumerate(case_scores[:, :2]):
            dx, dy = offsets[j % len(offsets)]
            ax.text(x + dx, y + dy, chr(65 + j), fontsize=11, fontweight="bold", ha="center", va="center")
    ax.set_title("BACNB Monte Carlo - State-Space Map by Similarity")
    ax.set_xlabel(f"PC1 of similarities ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 of similarities ({explained[1] * 100:.1f}%)")
    ax.legend(title="Functional state", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, markerscale=1.5)
    fig.savefig(outdir / "BACNB_Fig14_MapaEspacoEstadosSimplex.png", bbox_inches="tight")
    plt.close(fig)



def fig_state_space_tetrahedron(outdir: Path, similaridades: pd.DataFrame, estado: pd.DataFrame, case_sim: pd.DataFrame | None, seed: int) -> None:
    """Represents P1-P4 weights as a composition in a tetrahedral simplex.

    Four similarity weights that sum to 1 have three degrees of freedom. The
    regular tetrahedron is therefore the natural geometric representation of the
    four-prototype compositional space. Vertices work as ideal attractors;
    internal points represent mixtures/transitions between prototypes.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    vertices = np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ], dtype=float) / math.sqrt(3.0)
    weights = similaridades[sim_cols].to_numpy(dtype=float)
    coords = weights @ vertices
    rng = np.random.default_rng(seed + 53)
    idx = rng.choice(len(similaridades), size=min(10000, len(similaridades)), replace=False)
    w_idx = weights[idx]
    order = np.argsort(w_idx, axis=1)
    top_idx = order[:, -1]
    second_idx = order[:, -2]
    dominance_gap = w_idx[np.arange(len(w_idx)), top_idx] - w_idx[np.arange(len(w_idx)), second_idx]
    proto_labels = np.array(["P1 regulation", "P2 variability", "P3 automation", "P4 economy"])
    labels = proto_labels[top_idx]
    labels = np.where(dominance_gap < 0.08, "Transition", labels)
    unique_labels = list(dict.fromkeys(labels))
    label_order = [
        "P1 regulation",
        "P2 variability",
        "P3 automation",
        "P4 economy",
        "Transition",
    ]
    unique_labels = [x for x in label_order if x in unique_labels]
    palette = {
        "P1 regulation": "#4C72B0",
        "P2 variability": "#DD8452",
        "P3 automation": "#55A868",
        "P4 economy": "#C44E52",
        "Transition": "#8172B3",
    }

    fig = plt.figure(figsize=(14.5, 9.2), constrained_layout=False)
    ax = fig.add_axes([0.02, 0.08, 0.72, 0.86], projection="3d")
    for label in unique_labels:
        mask = labels == label
        pts = coords[idx][mask]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=9, alpha=0.28, color=palette[label], label=label, depthshade=False)

    # Manual geometric frame: gives a sense of depth without imposing a numeric scale.
    frame_min, frame_max = -0.70, 0.70
    frame_edges = [
        ((frame_min, frame_min, frame_min), (frame_max, frame_min, frame_min)),
        ((frame_min, frame_min, frame_min), (frame_min, frame_max, frame_min)),
        ((frame_min, frame_min, frame_min), (frame_min, frame_min, frame_max)),
        ((frame_max, frame_min, frame_min), (frame_max, frame_max, frame_min)),
        ((frame_max, frame_min, frame_min), (frame_max, frame_min, frame_max)),
        ((frame_min, frame_max, frame_min), (frame_max, frame_max, frame_min)),
        ((frame_min, frame_max, frame_min), (frame_min, frame_max, frame_max)),
        ((frame_min, frame_min, frame_max), (frame_max, frame_min, frame_max)),
        ((frame_min, frame_min, frame_max), (frame_min, frame_max, frame_max)),
    ]
    for a, b in frame_edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#D7D7D7", linewidth=0.8, alpha=0.42, zorder=0)

    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for a, b in edges:
        ax.plot([vertices[a, 0], vertices[b, 0]], [vertices[a, 1], vertices[b, 1]], [vertices[a, 2], vertices[b, 2]], color="black", linewidth=1.4, alpha=0.65)
    label_offsets = {
        1: np.array([0.065, 0.065, 0.070]),
        2: np.array([0.125, -0.110, 0.080]),
        3: np.array([-0.130, 0.135, 0.080]),
        4: np.array([-0.080, -0.080, 0.125]),
    }
    for i, pid in enumerate(sorted(PROFILE_COLUMNS)):
        label_pos = vertices[i] + label_offsets.get(pid, np.zeros(3))
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            PROFILE_SHORT[pid],
            fontsize=16,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
        )

    if case_sim is not None and len(case_sim):
        case_coords = case_sim[sim_cols].to_numpy(dtype=float) @ vertices
        ax.scatter(case_coords[:, 0], case_coords[:, 1], case_coords[:, 2], s=34, color="white", edgecolor="black", linewidth=0.9, marker="o", depthshade=False, label="_nolegend_")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(frame_min, frame_max)
    ax.set_ylim(frame_min, frame_max)
    ax.set_zlim(frame_min, frame_max)
    ax.view_init(elev=22, azim=38)
    ax.grid(False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.0, markersize=5))
    legend_labels.append("Ideal anchors")
    legend_ax = fig.add_axes([0.76, 0.45, 0.22, 0.34])
    legend_ax.axis("off")
    legend = legend_ax.legend(
        handles,
        legend_labels,
        title="Dominant proximity",
        loc="upper left",
        frameon=True,
        markerscale=2.1,
        borderpad=0.7,
        labelspacing=0.55,
        handletextpad=0.55,
        fontsize=12,
        title_fontsize=13,
    )
    legend.get_frame().set_linewidth(1.1)
    fig.savefig(outdir / "BACNB_Fig18_TetraedroAtratoresEstado.png", bbox_inches="tight")
    plt.close(fig)

def fig_state_categories(outdir: Path, estado: pd.DataFrame) -> None:
    counts = estado["estado_funcional"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(17, max(8, 0.55 * len(counts) + 4)), constrained_layout=True)
    bars = ax.barh(np.arange(len(counts)), counts.values, color=sns.color_palette("tab20", len(counts)), edgecolor="black", linewidth=1.0)
    total = counts.sum()
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels([pretty_state_label(x) for x in counts.index])
    ax.set_xlabel("Number of synthetic agents")
    ax.set_ylabel("State-space category")
    ax.set_title("BACNB Monte Carlo - State-Space Categories")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + total * 0.002, bar.get_y() + bar.get_height() / 2, f"{val} ({100 * val / total:.1f}%)", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, counts.max() * 1.18)
    fig.savefig(outdir / "BACNB_Fig15_CategoriasEspacoEstados.png", bbox_inches="tight")
    plt.close(fig)


def fig_synthetic_state_space(outdir: Path, case_sim: pd.DataFrame, case_estado: pd.DataFrame) -> None:
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    plot = case_sim.set_index("case_id")[sim_cols] * 100
    plot.columns = [PROFILE_SHORT[pid] for pid in sorted(PROFILE_COLUMNS)]
    fig, ax = plt.subplots(figsize=(16, 8.5), constrained_layout=True)
    x = np.arange(len(plot))
    width = 0.18
    for i, col in enumerate(plot.columns):
        ax.bar(x + (i - 1.5) * width, plot[col], width=width, label=col, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot.index, rotation=18, ha="right")
    ax.set_ylabel("Functional similarity (%)")
    ax.set_xlabel("Synthetic case")
    ax.set_ylim(0, 105)
    ax.set_title("BACNB Monte Carlo - Synthetic Cases in the State Space")
    ax.legend(title="Prototype", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.savefig(outdir / "BACNB_Fig16_CasosSinteticosNoEspacoEstados.png", bbox_inches="tight")
    plt.close(fig)


def fig_pca_with_prototypes(outdir: Path, x_def: pd.DataFrame, clusters: pd.Series, prototipos: dict[str, Any], explained: np.ndarray, seed: int) -> None:
    scores, explained2, vt = pca_scores(x_def.to_numpy(dtype=float), n_components=2)
    rng = np.random.default_rng(seed + 47)
    idx = rng.choice(len(x_def), size=min(12000, len(x_def)), replace=False)
    plot = pd.DataFrame({"PC1": scores[idx, 0], "PC2": scores[idx, 1], "Profile": clusters.iloc[idx].map(lambda x: PROFILE_SHORT.get(int(x), str(x))).to_numpy()})
    cent = prototipos["metric_centroids"][x_def.columns].to_numpy(dtype=float)
    cent_scores = (cent - x_def.to_numpy(dtype=float).mean(axis=0)) @ vt[:2].T
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    sns.scatterplot(data=plot, x="PC1", y="PC2", hue="Profile", s=15, alpha=0.25, linewidth=0, palette="tab10", ax=ax)
    ax.scatter(cent_scores[:, 0], cent_scores[:, 1], s=260, c="black", marker="X", label="Prototypes P1-P4", zorder=5)
    for xy, pid in zip(cent_scores, prototipos["metric_centroids"].index):
        ax.text(xy[0], xy[1], PROFILE_SHORT[int(pid)], fontsize=15, fontweight="bold", ha="left", va="bottom")
    ax.set_title("BACNB Monte Carlo - PCA with Functional Prototypes")
    ax.set_xlabel(f"PC1 ({explained2[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({explained2[1] * 100:.1f}% of variance)")
    ax.legend(title="Reference", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.savefig(outdir / "BACNB_Fig17_PCAComPrototipos.png", bbox_inches="tight")
    plt.close(fig)


def write_synthetic_report(outdir: Path, cases_z: pd.DataFrame, case_sim: pd.DataFrame, case_estado: pd.DataFrame, case_constructs: pd.DataFrame, case_tests: pd.DataFrame) -> None:
    lines = []
    add = lines.append
    add("BACNB SYNTHETIC CASE REPORT")
    add("=" * 88)
    add("")
    for i, case_id in enumerate(cases_z["case_id"]):
        sim_row = case_sim.iloc[i]
        estado_row = case_estado.iloc[i]
        constructs = case_constructs.drop(columns=["case_id"]).iloc[i]
        tests = case_tests.drop(columns=["case_id"]).iloc[i]
        x_row = cases_z.drop(columns=["case_id"]).iloc[i]
        add(case_id)
        add("-" * 88)
        add(interpretar_perfil_individual(x_row, sim_row, estado_row, constructs, tests))
        add("")
        sim_txt = "; ".join(f"{PROFILE_SHORT[pid]}={sim_row[PROFILE_COLUMNS[pid]]:.2f}" for pid in sorted(PROFILE_COLUMNS))
        add(f"Similarities: {sim_txt}")
        add(f"Functional state: {estado_row['estado_funcional']}")
        add(f"Subtype: {estado_row['subtipo_estado']}")
        add("")
    (outdir / "BACNB_SYNTHETIC_CASE_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


def load_individual_metrics(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    return pd.read_csv(path)


def process_individual_file(
    outdir: Path,
    individual_file: str,
    transformer: dict[str, Any],
    distance_prototipos: dict[str, Any],
    var_info: pd.DataFrame,
    tau: float,
    method: str,
    referencia_estado: dict[str, Any],
    similarity_space: str = "metrics",
) -> None:
    path = Path(individual_file)
    if not path.exists():
        raise FileNotFoundError(f"Individual file not found: {path}")
    metrics = load_individual_metrics(path)
    missing = [spec.col for spec in FEATURES if spec.col not in metrics.columns]
    if missing:
        raise ValueError(f"Individual file missing required metrics: {missing}")
    if "agent_id" not in metrics.columns:
        metrics.insert(0, "agent_id", [f"individuo_{i+1}" for i in range(len(metrics))])
    x_ind = apply_deficit_transformer(metrics, transformer)
    ind_constructs = compute_emergent_scores(x_ind, var_info)
    ind_tests = aggregate_by_test(x_ind)
    ind_similarity_space = build_similarity_space(x_ind, ind_constructs, similarity_space)
    ind_dist = calcular_distancias_perfis(ind_similarity_space, distance_prototipos, metodo=method, agent_ids=metrics["agent_id"])
    ind_sim = calcular_similaridade_funcional(ind_dist, tau=tau)
    ind_vector = calcular_vetor_estado_unitario(ind_sim)
    ind_estado = categorizar_estado_funcional(ind_sim, ind_dist, x_ind, ind_constructs, ind_tests, referencia_estado)

    x_out = x_ind.copy()
    x_out.insert(0, "agent_id", metrics["agent_id"].to_numpy())
    x_out.to_csv(outdir / "individual_deficit_z_metrics_bacnb.csv", index=False, encoding="utf-8-sig")
    ind_dist.to_csv(outdir / "individuos_distancias_perfis_bacnb.csv", index=False, encoding="utf-8-sig")
    ind_sim.to_csv(outdir / "individuos_similaridade_funcional_perfis_bacnb.csv", index=False, encoding="utf-8-sig")
    ind_vector.to_csv(outdir / "individuos_vetor_estado_bacnb.csv", index=False, encoding="utf-8-sig")
    ind_estado.to_csv(outdir / "individuos_categorizacao_espaco_estados_bacnb.csv", index=False, encoding="utf-8-sig")

    lines = ["BACNB INDIVIDUAL REPORT", "=" * 88, ""]
    for i in range(len(metrics)):
        lines.append(str(metrics.iloc[i]["agent_id"]))
        lines.append("-" * 88)
        lines.append(interpretar_perfil_individual(x_ind.iloc[i], ind_sim.iloc[i], ind_estado.iloc[i], ind_constructs.iloc[i], ind_tests.iloc[i]))
        lines.append("")
    (outdir / "BACNB_INDIVIDUAL_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


def save_csvs(outdir: Path, lat: pd.DataFrame, metrics: pd.DataFrame, x_def: pd.DataFrame, var_info: pd.DataFrame, emergent: pd.DataFrame, miyake: pd.DataFrame, clusters: pd.Series, centroids: pd.DataFrame, k_criteria: pd.DataFrame, var_criteria: pd.DataFrame, cluster_labels: dict[int, str] | None = None) -> None:
    lat_out = lat.copy()
    lat_out.insert(0, "agent_id", metrics["agent_id"].to_numpy())
    lat_out.to_csv(outdir / "agentes_microparametros_latentes.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(outdir / "agentes_metricas_observadas_bacnb.csv", index=False, encoding="utf-8-sig")
    x_def2 = x_def.copy()
    x_def2.insert(0, "agent_id", metrics["agent_id"].to_numpy())
    x_def2.to_csv(outdir / "agent_deficit_z_metrics_bacnb.csv", index=False, encoding="utf-8-sig")
    var_info.to_csv(outdir / "emergent_variable_constructs_bacnb.csv", index=False, encoding="utf-8-sig")
    emergent2 = emergent.copy()
    emergent2.insert(0, "agent_id", metrics["agent_id"].to_numpy())
    emergent2.to_csv(outdir / "agent_emergent_construct_scores_bacnb.csv", index=False, encoding="utf-8-sig")
    miyake2 = miyake.copy()
    miyake2.insert(0, "agent_id", metrics["agent_id"].to_numpy())
    miyake2.to_csv(outdir / "agentes_scores_miyake_teorico_bacnb.csv", index=False, encoding="utf-8-sig")
    cl = pd.DataFrame({"agent_id": metrics["agent_id"], "cluster": clusters, "perfil_id": clusters})
    if cluster_labels:
        cl["perfil_nome"] = cl["perfil_id"].map(lambda x: cluster_labels.get(int(x), profile_label(int(x))))
    cl.to_csv(outdir / "agentes_clusters_bacnb.csv", index=False, encoding="utf-8-sig")
    centroids.to_csv(outdir / "deficit_z_cluster_centroids_bacnb.csv", encoding="utf-8-sig")
    k_criteria.to_csv(outdir / "criterio_k_clusters_agentes_bacnb.csv", index=False, encoding="utf-8-sig")
    var_criteria.to_csv(outdir / "variable_construct_k_criteria_bacnb.csv", index=False, encoding="utf-8-sig")


def fig_corr_heatmap(outdir: Path, x_def: pd.DataFrame, var_info: pd.DataFrame) -> None:
    order = var_info.sort_values(["constructo_emergente_id", "teste", "variavel"])["variavel"].tolist()
    labels = [next(spec.short for spec in FEATURES if spec.col == col) for col in order]
    corr = x_def[order].corr()
    fig, ax = plt.subplots(figsize=(20, 17), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "correlation"},
        ax=ax,
    )
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title("BACNB Monte Carlo - Observed Metric Correlation")
    fig.savefig(outdir / "BACNB_Fig1_CorrelacaoMetricas.png", bbox_inches="tight")
    plt.close(fig)


def fig_dendrogram(outdir: Path, z_link: np.ndarray, var_info: pd.DataFrame) -> None:
    labels = [spec.short for spec in FEATURES]
    fig, ax = plt.subplots(figsize=(22, 11), constrained_layout=True)
    dendrogram(z_link, labels=labels, leaf_rotation=45, leaf_font_size=13, color_threshold=None, ax=ax)
    ax.set_title("BACNB Monte Carlo - Metric Dendrogram (Emergent Constructs)")
    ax.set_ylabel("Correlation distance")
    ax.set_xlabel("Observable metrics from the four tests")
    fig.savefig(outdir / "BACNB_Fig2_DendrogramaMetricas.png", bbox_inches="tight")
    plt.close(fig)


def fig_pca_clusters(outdir: Path, scores: np.ndarray, clusters: pd.Series, explained: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed + 29)
    idx = rng.choice(scores.shape[0], size=min(10000, scores.shape[0]), replace=False)
    plot = pd.DataFrame(
        {
            "PC1": scores[idx, 0],
            "PC2": scores[idx, 1],
            "Cluster": clusters.iloc[idx].astype(str).to_numpy(),
        }
    )
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    sns.scatterplot(data=plot, x="PC1", y="PC2", hue="Cluster", s=18, alpha=0.45, linewidth=0, palette="tab10", ax=ax)
    ax.set_title("BACNB Monte Carlo - Synthetic Agents in PCA Space")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% of variance)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    fig.savefig(outdir / "BACNB_Fig3_PCAClustersAgentes.png", bbox_inches="tight")
    plt.close(fig)


def fig_cluster_centroids(outdir: Path, centroids: pd.DataFrame, cluster_labels: dict[int, str]) -> None:
    rename = {spec.col: spec.short for spec in FEATURES}
    mat = centroids.rename(columns=rename).copy()
    mat.index = [cluster_labels.get(int(i), str(i)) for i in mat.index]
    vmin, vmax = symmetric_limits(mat)
    fig, ax = plt.subplots(figsize=(24, max(8, 1.0 * len(mat) + 4)), constrained_layout=True)
    sns.heatmap(
        mat,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "oriented functional z"},
        ax=ax,
    )
    ax.set_title("BACNB Monte Carlo - Profile Centroids by Metric")
    ax.set_xlabel("Observable metric (higher = worse after orientation)")
    ax.set_ylabel("Emergent functional profile")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.savefig(outdir / "BACNB_Fig4_CentroidesClustersMetricas.png", bbox_inches="tight")
    plt.close(fig)


def fig_emergent_vs_miyake(outdir: Path, emergent: pd.DataFrame, miyake: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = pd.concat([emergent, miyake], axis=1).corr().loc[emergent.columns, miyake.columns]
    p_rows: list[dict[str, Any]] = []
    for emergent_col in emergent.columns:
        for miyake_col in miyake.columns:
            r, p_value = stats.pearsonr(emergent[emergent_col].to_numpy(dtype=float), miyake[miyake_col].to_numpy(dtype=float))
            p_rows.append(
                {
                    "constructo_emergente": emergent_col,
                    "constructo_miyake": miyake_col,
                    "r_pearson": float(r),
                    "p_value": float(p_value),
                    "p_apa": format_p_apa(float(p_value)),
                    "significativo_p05": bool(p_value < ALPHA),
                }
            )
    p_table = pd.DataFrame(p_rows)
    fig, ax = plt.subplots(figsize=(14, max(7, 1.2 * len(corr))), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "correlation"},
        ax=ax,
    )
    ax.set_title("BACNB Monte Carlo - Emergent Constructs vs. Miyake")
    ax.set_xlabel("Theoretical Miyake constructs (pre-specified proxies)")
    ax.set_ylabel("Data-driven emergent constructs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.savefig(outdir / "BACNB_Fig5_EmergentesVsMiyake.png", bbox_inches="tight")
    plt.close(fig)
    return corr, p_table


def fig_cluster_constructs(outdir: Path, emergent: pd.DataFrame, clusters: pd.Series, cluster_labels: dict[int, str]) -> pd.DataFrame:
    data = emergent.copy()
    data["cluster"] = clusters.to_numpy()
    cent = data.groupby("cluster").mean()
    cent.index = [cluster_labels.get(int(i), str(i)) for i in cent.index]
    vmin, vmax = symmetric_limits(cent)
    fig, ax = plt.subplots(figsize=(18, max(8, len(cent) * 1.1 + 3)), constrained_layout=True)
    sns.heatmap(
        cent,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "oriented functional z"},
        ax=ax,
    )
    ax.set_title("BACNB Monte Carlo - Profiles in Emergent Constructs")
    ax.set_xlabel("Emergent construct")
    ax.set_ylabel("Functional profile")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.savefig(outdir / "BACNB_Fig6_PerfisConstructosEmergentes.png", bbox_inches="tight")
    plt.close(fig)
    return cent


def fig_cluster_tests(outdir: Path, test_scores: pd.DataFrame, clusters: pd.Series, cluster_labels: dict[int, str]) -> pd.DataFrame:
    data = test_scores.copy()
    data["cluster"] = clusters.to_numpy()
    cent = data.groupby("cluster").mean()
    cent.index = [cluster_labels.get(int(i), str(i)) for i in cent.index]
    vmin, vmax = symmetric_limits(cent)
    fig, ax = plt.subplots(figsize=(14, max(7, len(cent) * 1.0 + 3)), constrained_layout=True)
    sns.heatmap(
        cent,
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "oriented functional z"},
        ax=ax,
    )
    ax.set_title("BACNB Monte Carlo - Functional Profile by Test")
    ax.set_xlabel("Test")
    ax.set_ylabel("Functional profile")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.savefig(outdir / "BACNB_Fig7_PerfisPorTeste.png", bbox_inches="tight")
    plt.close(fig)
    return cent


def fig_k_criteria(outdir: Path, k_criteria: pd.DataFrame, selected_k: int) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax1.plot(k_criteria["k"], k_criteria["centroid_silhouette"], marker="o", linewidth=3, label="Centroid silhouette")
    ax1.axvline(selected_k, color="red", linestyle="--", linewidth=2, label=f"Selected K = {selected_k}")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Approximate silhouette")
    ax2 = ax1.twinx()
    ax2.plot(k_criteria["k"], k_criteria["menor_cluster"] * 100, marker="s", linewidth=2, color="gray", label="Smallest cluster (%)")
    ax2.set_ylabel("Smallest cluster (%)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=True)
    ax1.set_title("BACNB Monte Carlo - K-Selection Audit")
    fig.savefig(outdir / "BACNB_Fig8_CriterioKClusters.png", bbox_inches="tight")
    plt.close(fig)


def fig_cluster_sizes(outdir: Path, clusters: pd.Series, cluster_labels: dict[int, str]) -> None:
    counts = clusters.value_counts().sort_index()
    labels = [cluster_labels.get(int(i), str(i)).replace("Cluster ", "C") for i in counts.index]
    fig, ax = plt.subplots(figsize=(17, 9.5), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, counts.values, color=sns.color_palette("tab10", len(counts)), edgecolor="black", linewidth=1.5)
    total = counts.sum()
    ax.set_ylim(0, counts.max() * 1.18)
    for b, v in zip(bars, counts.values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + counts.max() * 0.015,
            f"{v}\n({100*v/total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_title("BACNB Monte Carlo - Functional Profile Sizes", pad=18)
    ax.set_ylabel("Number of synthetic agents")
    ax.set_xlabel("Profile")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.savefig(outdir / "BACNB_Fig9_TamanhoClusters.png", bbox_inches="tight")
    plt.close(fig)


def fig_latent_audit(outdir: Path, lat: pd.DataFrame, x_def: pd.DataFrame) -> pd.DataFrame:
    latent_labels = {k: v for k, v in LATENT_LABELS.items()}
    feature_labels = {spec.col: spec.short for spec in FEATURES}
    corr = pd.concat([lat, x_def], axis=1).corr().loc[lat.columns, x_def.columns]
    corr.index = [latent_labels.get(i, i) for i in corr.index]
    corr.columns = [feature_labels.get(c, c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(22, 10), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "correlation"},
        ax=ax,
    )
    ax.set_title("BACNB Monte Carlo - Latent vs. Observed Metric Audit")
    ax.set_xlabel("Observable metrics oriented as deficit")
    ax.set_ylabel("Generative microparameters")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.savefig(outdir / "BACNB_Fig10_AuditoriaLatenteObservada.png", bbox_inches="tight")
    plt.close(fig)
    return corr


def write_report(
    outdir: Path,
    args: argparse.Namespace,
    selected_k: int,
    var_info: pd.DataFrame,
    var_criteria: pd.DataFrame,
    k_criteria: pd.DataFrame,
    cluster_labels: dict[int, str],
    clusters: pd.Series,
    centroids: pd.DataFrame,
    emergent_miyake_corr: pd.DataFrame,
    construct_centroids: pd.DataFrame,
    test_centroids: pd.DataFrame,
    emergent_miyake_pvalues: pd.DataFrame,
    similaridades: pd.DataFrame,
    sobreposicao: pd.DataFrame,
    estado_funcional: pd.DataFrame,
    prototipos: dict[str, Any],
    case_estado: pd.DataFrame | None = None,
) -> None:
    lines = []
    add = lines.append
    add("BACNB IN SILICO EXPERIMENT - INTEGRATED MONTE CARLO")
    add("=" * 88)
    add("")
    add(f"Date: {DATE_TAG}")
    add(f"N synthetic agents: {args.n}")
    add(f"Seed: {args.seed}")
    add(f"Global observational noise: {args.noise:.3f}")
    se_max = max_standard_error_proportion(args.n)
    add(f"Maximum standard error for proportions at p=0.50: {se_max * 100:.3f} percentage point.")
    add("Integrated tests in the same agent: SART, SST, Flanker, and Digit Span.")
    add("")
    add("1. WHAT THIS MONTE CARLO IS FOR")
    add("-" * 88)
    add(textwrap.fill(
        "This in silico experiment audits whether BACNB can transform plausible "
        "behavioral differences into observable metrics, whether those metrics cluster "
        "into emergent constructs, and whether the same simulated agents produce "
        "consistent profiles across the four tests. These profiles are treated as "
        "idealized clinical-functional prototypes, not as diagnostic classes. The "
        "simulation does not create clinical norms, does not prove diagnostic validity, "
        "and does not demonstrate real biology. It tests a conditional proposition: "
        "given transparent generative rules and controlled noise, can the battery "
        "recover stable functional patterns?", width=100))
    add("")
    add("2. HOW THE BLACK BOX WAS OPENED")
    add("-" * 88)
    add(textwrap.fill(
        "Each agent received continuous microparameters: general executive factor, "
        "slowness/conservative strategy, motor impulsivity, stop-cancellation deficit, "
        "attentional lapses, temporal variability, conflict susceptibility, verbal-memory "
        "deficit, and rigidity/automation. These microparameters generate all four tests "
        "for the same agent. The clustering step, however, does not see these parameters: "
        "it receives only the simulated observable task metrics.", width=100))
    add("")
    add(textwrap.fill(
        "Anti-circularity rule: microparameters -> observable metrics -> deficit z-score "
        "-> emergent constructs based on variable correlations -> agent clustering to "
        "generate functional prototypes -> dimensional similarity to the prototypes. "
        "The Miyake model enters only afterward, as a pre-specified external comparison. "
        "Therefore, emergent constructs are not forced to become Inhibition, Working "
        "Memory, or Flexibility.", width=100))
    add("")
    add("3. EMERGENT CONSTRUCTS FROM VARIABLES")
    add("-" * 88)
    best_var = var_criteria.sort_values("silhouette_distancia_correlacao", ascending=False).iloc[0]
    add(f"Selected variable-construct K: {var_info['constructo_emergente_id'].nunique()} (best tested criterion: K={int(best_var['k'])}, silhouette={best_var['silhouette_distancia_correlacao']:.3f}).")
    add("")
    for cname, group in var_info.groupby("constructo_emergente"):
        vars_txt = ", ".join(f"{r.rotulo_curto} [{r.teste}]" for r in group.itertuples())
        add(f"- {cname}: {vars_txt}")
    add("")
    add("4. FUNCTIONAL PROTOTYPES GENERATED BY AGENT CLUSTERING")
    add("-" * 88)
    add(f"Automatically selected agent K: {selected_k}")
    add("Criterion: highest approximate centroid silhouette among solutions with smallest cluster >= 3%.")
    add(textwrap.fill(
        "Clustering is still used, but only to generate and audit the P1-P4 prototypes. "
        "In individual interpretation, a person does not rigidly 'belong' to a cluster; "
        "they occupy a position in multivariate space and show degrees of functional "
        "similarity to the ideal prototypes.", width=100))
    add("")
    add(k_criteria.to_string(index=False, formatters={
        "inertia": "{:.2f}".format,
        "centroid_silhouette": "{:.3f}".format,
        "menor_cluster": "{:.3f}".format,
        "maior_cluster": "{:.3f}".format,
        "inertia_delta_rel": lambda x: "" if pd.isna(x) else f"{x:.3f}",
    }))
    add("")
    counts = clusters.value_counts().sort_index()
    for cid, count in counts.items():
        label = cluster_labels[int(cid)]
        pct = 100 * count / len(clusters)
        top = centroids.loc[cid].sort_values(ascending=False).head(6)
        bottom = centroids.loc[cid].sort_values(ascending=True).head(3)
        top_txt = "; ".join(f"{next(s.short for s in FEATURES if s.col == k)}={v:+.2f}" for k, v in top.items())
        bottom_txt = "; ".join(f"{next(s.short for s in FEATURES if s.col == k)}={v:+.2f}" for k, v in bottom.items())
        add(f"{label}: n={count} ({pct:.1f}%).")
        add(f"  Highest elevations: {top_txt}.")
        add(f"  Lowest elevations/protections: {bottom_txt}.")
    add("")
    add("5. EMERGENT EXECUTIVE FUNCTIONS BY PROTOTYPE")
    add("-" * 88)
    add(construct_centroids.to_string(float_format=lambda x: f"{x:+.2f}"))
    add("")
    add("6. OVERLAP OF THE FOUR TESTS BY PROTOTYPE")
    add("-" * 88)
    add(test_centroids.to_string(float_format=lambda x: f"{x:+.2f}"))
    add("")
    add("7. COMPARISON WITH MIYAKE")
    add("-" * 88)
    add(textwrap.fill(
        "The matrix below correlates data-driven emergent constructs with three "
        "pre-specified theoretical scores: Inhibitory Control, Working Memory, and "
        "Cognitive Flexibility. This does not force the battery to fit Miyake; it only "
        "measures the similarity between the emergent structure and that theory.", width=100))
    add("")
    add(emergent_miyake_corr.to_string(float_format=lambda x: f"{x:+.2f}"))
    add("")
    add("P-values for emergent vs. Miyake correlations")
    add("(significance criterion: alpha = .05; APA notation, with p < .001 when applicable)")
    if len(emergent_miyake_pvalues):
        p_view = emergent_miyake_pvalues.sort_values(["constructo_emergente", "constructo_miyake"]).copy()
        p_view["r_pearson"] = p_view["r_pearson"].map(lambda x: f"{x:+.3f}")
        p_view["significativo_p05"] = p_view["significativo_p05"].map(lambda x: "yes" if x else "no")
        add(p_view[["constructo_emergente", "constructo_miyake", "r_pearson", "p_apa", "significativo_p05"]].to_string(index=False))
    add("")
    add(textwrap.fill(
        "With a massive N, very small p-values are expected even for modest effects. "
        "Statistical significance is therefore reported using the p < .05 standard, but "
        "substantive interpretation prioritizes r magnitude, geometric stability, "
        "prototype coherence, and neuropsychological plausibility.", width=100))
    add("")
    add(textwrap.fill(
        "Critical interpretation: in BACNB, Inhibitory Control tends to be well covered "
        "by SART/SST/Flanker, and Working Memory is covered mainly by Digit Span. "
        "Cognitive Flexibility remains underidentified because Wisconsin/BCST and Corsi "
        "were removed. The adapted SART captures rigidity/automation, but this is not "
        "equivalent to a classic rule-switching test.", width=100))
    add("")
    add("8. FUNCTIONAL STATE VECTOR")
    add("-" * 88)
    sim_cols = [PROFILE_COLUMNS[pid] for pid in sorted(PROFILE_COLUMNS)]
    sim_summary = similaridades[sim_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    sim_summary.index = [PROFILE_SHORT[pid] for pid in sorted(PROFILE_COLUMNS)]
    add((100 * sim_summary[["mean", "std", "25%", "50%", "75%"]]).to_string(float_format=lambda x: f"{x:.2f}%"))
    add("")
    add(textwrap.fill(
        "The model's main output is the functional state vector v = [P1, P2, P3, P4]. "
        "Each component expresses the agent's percentage approximation to the corresponding "
        "attractor/prototype, and the four percentages sum to 100%. This vector is a "
        "compositional coordinate in the tetrahedral simplex: it is not a diagnosis, "
        "natural class, or rigid membership assignment. The predominant profile is only "
        "the largest vector component; the scientifically richer information is in the "
        "full composition and in the margin between components.", width=100))
    add("")
    add(textwrap.fill(
        "Mathematically, the vector is obtained in three steps: first, the agent's distance "
        "to the four prototypes is computed in multivariate space; then distances are "
        "converted into weights through a softmax transformation over negative energy, "
        "exp(-d^2/2tau^2); finally, weights are multiplied by 100. Agents close to a "
        "single attractor therefore have an almost pure vector, whereas intermediate "
        "agents show proportional mixtures between attractors.", width=100))
    add("")
    add("9. STATE-SPACE CATEGORIZATION")
    add("-" * 88)
    state_counts = estado_funcional["estado_funcional"].value_counts()
    add(state_counts.to_string())
    add("")
    add(textwrap.fill(
        "State-space categorization describes the functional region occupied by the agent "
        "relative to the P1-P4 prototypes. This layer distinguishes prototypical, mixed, "
        "overlapping, asymmetric/dissociated, and poorly represented states. It should "
        "always be read together with similarity weights, altered constructs, and preserved "
        "tests.", width=100))
    add("")
    add("10. PROFILES AS IDEAL MODELS, NOT DIAGNOSTIC CATEGORIES")
    add("-" * 88)
    add(textwrap.fill(
        "BACNB interprets the four profiles as idealized clinical-functional models. They "
        "are not rigid natural types, diagnostic categories, or clinical norms. They work "
        "as mathematical references for locating individual patterns in a multivariate "
        "performance space. An individual may therefore show different degrees of similarity "
        "to multiple profiles, including asymmetric configurations and gradual transitions "
        "between prototypical attractors in the state space.", width=100))
    add("")
    add("11. SYNTHETIC CASES FOR INTERPRETIVE VALIDATION")
    add("-" * 88)
    if case_estado is not None and len(case_estado):
        for row in case_estado.itertuples(index=False):
            add(f"{row.case_id}: {row.estado_funcional} | {row.subtipo_estado}")
    else:
        add("Synthetic cases were not executed in this run.")
    add("")
    add(textwrap.fill(
        "The synthetic cases audit only the four ideal P1-P4 anchors. They verify whether "
        "the state space recognizes prototypical regions of preserved control, lapses/"
        "variability, prepotent automation, and global control economy without introducing "
        "specific clinical cases.", width=100))
    add("")
    add("12. LIMITS OF INDIVIDUAL INTERPRETATION")
    add("-" * 88)
    add(textwrap.fill(
        "Monte Carlo results do not constitute clinical norms, diagnostic validity, or "
        "biological evidence for the profiles. They provide a formal in silico reference "
        "for auditing BACNB internal coherence and guiding dimensional interpretation of "
        "individual patterns. Clinical application requires pilot sampling, test-retest "
        "reliability, empirical distribution analysis by age/education, and convergent/"
        "divergent validation.", width=100))
    add("")
    add("13. WHAT THIS RESULT PROVES AND DOES NOT PROVE")
    add("-" * 88)
    add(textwrap.fill(
        "It proves that the integrated pipeline is mathematically coherent, that the same "
        "agents can be simulated across the four tests, that emergent constructs can be "
        "extracted without using Miyake as a label, that prototypes can be described by "
        "patterns of overlap across tests, and that functional similarity allows "
        "dimensional interpretation instead of rigid assignment.", width=100))
    add("")
    add(textwrap.fill(
        "It does not prove clinical validity, diagnostic sensitivity, biological existence "
        "of the prototypes, or superiority over Miyake. To turn this into empirical science, "
        "real data must be collected, test-retest stability must be tested, factor/cluster "
        "analyses must be run in human participants, and competing models must be compared.", width=100))
    add("")
    add("14. GENERATED FILES")
    add("-" * 88)
    for p in sorted(outdir.iterdir()):
        add(f"- {p.name}")
    add("")

    (outdir / "BACNB_MONTE_CARLO_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    run_synthetic_cases = not args.no_synthetic_cases
    outdir = Path(args.outdir) if args.outdir else ROOT / "outputs" / f"BACNB_MC_{DATE_TAG}_N{args.n}_seed{args.seed}"
    ensure_dir(outdir)

    lat, metrics = simulate_bacnb(args.n, args.seed, args.noise)
    if args.z_reference_file:
        transformer = json.loads(Path(args.z_reference_file).read_text(encoding="utf-8"))
    else:
        transformer = fit_deficit_transformer(metrics)
    x_def = apply_deficit_transformer(metrics, transformer)
    save_json(outdir / "deficit_z_reference_bacnb.json", transformer)

    var_info, z_link, _var_distance, var_criteria = cluster_variables(x_def)
    emergent = compute_emergent_scores(x_def, var_info)
    miyake = compute_miyake_scores(x_def)
    test_scores = aggregate_by_test(x_def)

    scores, explained, _vt = pca_scores(x_def.to_numpy(), n_components=min(10, x_def.shape[1]))
    n_pc = int(np.searchsorted(np.cumsum(explained), 0.82) + 1)
    n_pc = max(3, min(n_pc, 8, scores.shape[1]))
    cluster_space = scores[:, :n_pc]
    selected_k, k_criteria = choose_agent_k(cluster_space, args.seed, args.max_k, args.sample_k)
    raw_labels, _centers, _inertia = run_kmeans(cluster_space, selected_k, np.random.default_rng(args.seed + 101), n_init=8, max_iter=80)
    labels, mapping = reorder_clusters(raw_labels, x_def)
    clusters = pd.Series(labels, name="cluster")
    centroids = x_def.copy()
    centroids["cluster"] = clusters
    centroids = centroids.groupby("cluster").mean()
    cluster_labels = label_agent_clusters(centroids)

    prototipos_metricas = calcular_prototipos_perfis(
        x_def=x_def,
        clusters=clusters,
        cluster_labels=cluster_labels,
        emergent_scores=emergent,
        test_scores=test_scores,
    )
    similarity_space = build_similarity_space(x_def, emergent, args.similarity_space)
    distance_prototipos = calcular_prototipos_perfis(
        x_def=similarity_space,
        clusters=clusters,
        cluster_labels=cluster_labels,
    )
    distancias = calcular_distancias_perfis(similarity_space, distance_prototipos, metodo=args.similarity_method, agent_ids=metrics["agent_id"])
    similaridades = calcular_similaridade_funcional(distancias, tau=args.similarity_tau)
    vetor_estado = calcular_vetor_estado_unitario(similaridades)
    sobreposicao = calcular_indice_sobreposicao(similaridades)
    referencia_estado_calibrada = calibrar_espaco_estados(similaridades, distancias, clusters, x_def, emergent, test_scores)
    save_json(outdir / "referencia_espaco_estados_bacnb.json", referencia_estado_calibrada)
    referencia_estado = json.loads(Path(args.state_reference_file).read_text(encoding="utf-8")) if args.state_reference_file else referencia_estado_calibrada
    estado_funcional = categorizar_estado_funcional(similaridades, distancias, x_def, emergent, test_scores, referencia_estado)

    save_csvs(outdir, lat, metrics, x_def, var_info, emergent, miyake, clusters, centroids, k_criteria, var_criteria, cluster_labels)

    proto_metricas = prototipos_metricas["metric_centroids"].copy()
    proto_metricas.insert(0, "perfil_nome", [cluster_labels.get(int(i), profile_label(int(i))) for i in proto_metricas.index])
    proto_metricas.to_csv(outdir / "profile_prototypes_deficit_z_metrics_bacnb.csv", encoding="utf-8-sig")
    proto_constructos = prototipos_metricas["construct_centroids"].copy()
    proto_constructos.insert(0, "perfil_nome", [cluster_labels.get(int(i), profile_label(int(i))) for i in proto_constructos.index])
    proto_constructos.to_csv(outdir / "profile_prototypes_constructs_bacnb.csv", encoding="utf-8-sig")
    proto_testes = prototipos_metricas["test_centroids"].copy()
    proto_testes.insert(0, "perfil_nome", [cluster_labels.get(int(i), profile_label(int(i))) for i in proto_testes.index])
    proto_testes.to_csv(outdir / "prototipos_perfis_por_teste_bacnb.csv", encoding="utf-8-sig")

    distancias.to_csv(outdir / "agentes_distancias_perfis_bacnb.csv", index=False, encoding="utf-8-sig")
    similaridades.to_csv(outdir / "agentes_similaridade_funcional_perfis_bacnb.csv", index=False, encoding="utf-8-sig")
    vetor_estado.to_csv(outdir / "agentes_vetor_estado_bacnb.csv", index=False, encoding="utf-8-sig")
    sobreposicao.to_csv(outdir / "agentes_sobreposicao_perfis_bacnb.csv", index=False, encoding="utf-8-sig")
    estado_funcional.to_csv(outdir / "agentes_categorizacao_espaco_estados_bacnb.csv", index=False, encoding="utf-8-sig")

    fig_corr_heatmap(outdir, x_def, var_info)
    fig_dendrogram(outdir, z_link, var_info)
    fig_pca_clusters(outdir, scores, clusters, explained, args.seed)
    fig_cluster_centroids(outdir, centroids, cluster_labels)
    emergent_miyake_corr, emergent_miyake_pvalues = fig_emergent_vs_miyake(outdir, emergent, miyake)
    construct_centroids = fig_cluster_constructs(outdir, emergent, clusters, cluster_labels)
    test_centroids = fig_cluster_tests(outdir, test_scores, clusters, cluster_labels)
    fig_k_criteria(outdir, k_criteria, selected_k)
    fig_cluster_sizes(outdir, clusters, cluster_labels)
    latent_corr = fig_latent_audit(outdir, lat, x_def)
    fig_similarity_profiles(outdir, similaridades, estado_funcional, args.seed)

    cases_z = pd.DataFrame()
    case_sim = pd.DataFrame()
    case_estado = pd.DataFrame()
    case_constructs = pd.DataFrame()
    case_tests = pd.DataFrame()
    if run_synthetic_cases:
        cases_z = criar_casos_sinteticos_validacao(prototipos_metricas)
        case_dist, case_sim, case_estado, case_constructs, case_tests = compute_case_outputs(
            cases_z=cases_z,
            distance_prototipos=distance_prototipos,
            var_info=var_info,
            tau=args.similarity_tau,
            method=args.similarity_method,
            referencia_estado=referencia_estado,
            similarity_space=args.similarity_space,
        )
        cases_z.to_csv(outdir / "casos_sinteticos_validacao_bacnb.csv", index=False, encoding="utf-8-sig")
        case_dist.to_csv(outdir / "casos_sinteticos_distancias_perfis_bacnb.csv", index=False, encoding="utf-8-sig")
        case_sim.to_csv(outdir / "casos_sinteticos_similaridade_perfis_bacnb.csv", index=False, encoding="utf-8-sig")
        calcular_vetor_estado_unitario(case_sim).to_csv(outdir / "casos_sinteticos_vetor_estado_bacnb.csv", index=False, encoding="utf-8-sig")
        case_estado.to_csv(outdir / "casos_sinteticos_categorizacao_espaco_estados_bacnb.csv", index=False, encoding="utf-8-sig")
        case_constructs.to_csv(outdir / "synthetic_case_constructs_bacnb.csv", index=False, encoding="utf-8-sig")
        write_synthetic_report(outdir, cases_z, case_sim, case_estado, case_constructs, case_tests)
        fig_synthetic_decomposition(outdir, case_sim)
        fig_synthetic_constructs(outdir, case_constructs)
        fig_synthetic_state_space(outdir, case_sim, case_estado)

    fig_state_space_simplex(outdir, similaridades, estado_funcional, case_sim if len(case_sim) else None, args.seed)
    fig_state_space_tetrahedron(outdir, similaridades, estado_funcional, case_sim if len(case_sim) else None, args.seed)
    fig_state_categories(outdir, estado_funcional)
    fig_pca_with_prototypes(outdir, x_def, clusters, prototipos_metricas, explained, args.seed)

    latent_corr.to_csv(outdir / "auditoria_latente_vs_observada.csv", encoding="utf-8-sig")
    emergent_miyake_corr.to_csv(outdir / "correlacao_emergentes_vs_miyake.csv", encoding="utf-8-sig")
    emergent_miyake_pvalues.to_csv(outdir / "correlacao_emergentes_vs_miyake_pvalores.csv", index=False, encoding="utf-8-sig")
    construct_centroids.to_csv(outdir / "emergent_construct_cluster_centroids.csv", encoding="utf-8-sig")
    test_centroids.to_csv(outdir / "centroides_clusters_por_teste.csv", encoding="utf-8-sig")

    metadata = {
        "date": DATE_TAG,
        "n": args.n,
        "seed": args.seed,
        "noise": args.noise,
        "selected_agent_k": selected_k,
        "selected_variable_constructs": int(var_info["constructo_emergente_id"].nunique()),
        "pca_components_for_clustering": n_pc,
        "pca_explained_used": float(np.sum(explained[:n_pc])),
        "output_dir": str(outdir),
        "cluster_label_map": cluster_labels,
        "anti_circularity": "Miyake scores are computed only after emergent variable clustering and agent clustering.",
        "profile_interpretation_mode": "dimensional_similarity_to_ideal_profiles",
        "hard_clusters_used_as": "prototype_generation_and_audit_only",
        "similarity_tau": args.similarity_tau,
        "similarity_method": args.similarity_method,
        "similarity_space": args.similarity_space,
        "z_deficit_reference_file": "deficit_z_reference_bacnb.json",
        "functional_similarity_file": "agentes_similaridade_funcional_perfis_bacnb.csv",
        "state_vector_file": "agentes_vetor_estado_bacnb.csv",
        "state_vector_definition": "v=[P1,P2,P3,P4], unit compositional vector of approximation to attractors; sum=1.00.",
        "state_vector_formula": "w_i = exp(-d_i^2/(2*tau^2)) / sum_j exp(-d_j^2/(2*tau^2)); P_i = w_i.",
        "state_vector_scale_note": "The textual concatenation uses semicolons between components to avoid ambiguity with decimal commas. Optional 0-10 or 0-100 scales are linear transformations that preserve proportions.",
        "overlap_index_file": "agentes_sobreposicao_perfis_bacnb.csv",
        "state_space_categorization": True,
        "state_space_categories": sorted(estado_funcional["estado_funcional"].unique().tolist()),
        "state_space_reference_file": "referencia_espaco_estados_bacnb.json",
        "state_space_categorization_file": "agentes_categorizacao_espaco_estados_bacnb.csv",
        "max_standard_error_proportion": max_standard_error_proportion(args.n),
        "max_standard_error_proportion_percentage_points": max_standard_error_proportion(args.n) * 100,
        "alpha_significance": ALPHA,
        "p_value_file": "correlacao_emergentes_vs_miyake_pvalores.csv",
        "broad_deficit_gate_enabled": True,
        "ood_detection_enabled": True,
        "synthetic_validation_cases_file": "casos_sinteticos_validacao_bacnb.csv" if run_synthetic_cases else "",
        "epistemic_note": (
            "BACNB uses idealized clinical-functional profiles as reference models. "
            "These profiles are not diagnostic categories or rigid natural types. "
            "Real individuals may occupy intermediate or asymmetric positions in the "
            "multivariate space, expressing different degrees of similarity to multiple profiles."
        ),
    }
    (outdir / "metadata_monte_carlo_bacnb.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.individual_file:
        process_individual_file(
            outdir=outdir,
            individual_file=args.individual_file,
            transformer=transformer,
            distance_prototipos=distance_prototipos,
            var_info=var_info,
            tau=args.similarity_tau,
            method=args.similarity_method,
            referencia_estado=referencia_estado,
            similarity_space=args.similarity_space,
        )

    write_report(
        outdir=outdir,
        args=args,
        selected_k=selected_k,
        var_info=var_info,
        var_criteria=var_criteria,
        k_criteria=k_criteria,
        cluster_labels=cluster_labels,
        clusters=clusters,
        centroids=centroids,
        emergent_miyake_corr=emergent_miyake_corr,
        construct_centroids=construct_centroids,
        test_centroids=test_centroids,
        emergent_miyake_pvalues=emergent_miyake_pvalues,
        similaridades=similaridades,
        sobreposicao=sobreposicao,
        estado_funcional=estado_funcional,
        prototipos=prototipos_metricas,
        case_estado=case_estado if len(case_estado) else None,
    )

    print(f"[OK] Monte Carlo BACNB completed: {outdir}")
    print(f"     N={args.n}, seed={args.seed}, noise={args.noise}, agent K={selected_k}, constructs={var_info['constructo_emergente_id'].nunique()}")


if __name__ == "__main__":
    main()
