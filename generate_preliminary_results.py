#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pydmd import HAVOK
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "slides" / "assets"

DT = 0.01
M = 100000
SVD_RANK = 11
DELAYS = 100
LONG_MULTIPLIER = 2
THRESHOLD_PROBABILITY = 0.09


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 11,
        }
    )


def generate_hr_data(t_eval: np.ndarray) -> np.ndarray:
    """Generate Hindmarsh-Rose neuron data in a chaotic bursting regime."""

    def hr_system(t: float, state: np.ndarray) -> list[float]:
        a = 1.0
        b = 3.0
        c = 1.0
        d = 5.0
        r = 0.006
        s = 4.0
        x0 = -1.6
        current = 3.25

        x, y, z = state
        x_dot = y - a * x**3 + b * x**2 - z + current
        y_dot = c - d * x**2 - y
        z_dot = r * (s * (x - x0) - z)
        return [x_dot, y_dot, z_dot]

    solution = solve_ivp(
        hr_system,
        [t_eval[0], t_eval[-1]],
        [-1.6, -10.0, 0.5],
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-12,
        method="LSODA",
    )

    return solution.y


def get_ind_burst_hr(
    x: np.ndarray, height: float = 0.0, min_distance: int = 100
) -> np.ndarray:
    """Detect burst onset events from the observed voltage trace."""

    above = (x > height).astype(int)
    onsets = np.where(np.diff(above) == 1)[0]
    if len(onsets) == 0:
        return onsets

    filtered = [onsets[0]]
    for idx in onsets[1:]:
        if idx - filtered[-1] > min_distance:
            filtered.append(idx)

    return np.array(filtered)


def get_time_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return (times >= start) & (times <= end)


def choose_burst_window(burst_times: np.ndarray, width: float, offset: int = 10) -> tuple[float, float]:
    center_index = min(offset, len(burst_times) - 1)
    center = float(burst_times[center_index])
    half_width = width / 2
    return center - half_width, center + half_width


def save_overview_figure(X: np.ndarray, t: np.ndarray, x: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(X[0], X[1], X[2], lw=0.8, color="#0b6e99")
    ax.set_title("Hindmarsh-Rose attractor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax = fig.add_subplot(1, 2, 2)
    mask = t <= 250
    ax.plot(t[mask], x[mask], color="#1d232f", lw=1.0)
    ax.set_title("Observed scalar signal x(t)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Membrane voltage")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "overview.png", bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_prediction_figure(
    t: np.ndarray,
    x: np.ndarray,
    reconstructed: np.ndarray,
    burst_times: np.ndarray,
    t_long: np.ndarray,
    x_long: np.ndarray,
    prediction: np.ndarray,
) -> None:
    recon_start, recon_end = choose_burst_window(burst_times, width=80.0, offset=10)
    pred_start, pred_end = choose_burst_window(burst_times + 1200.0, width=80.0, offset=12)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    recon_mask = get_time_mask(t, recon_start, recon_end)
    axes[0].plot(t[recon_mask], x[recon_mask], label="Truth", color="#1d232f", lw=1.1)
    axes[0].plot(
        t[recon_mask],
        reconstructed[recon_mask],
        label="HAVOK reconstruction",
        color="#d1495b",
        lw=1.1,
        alpha=0.95,
    )
    axes[0].set_title("Reconstruction on a burst-rich window")
    axes[0].set_ylabel("x(t)")
    axes[0].legend(loc="upper right")

    pred_mask = get_time_mask(t_long, pred_start, pred_end)
    axes[1].plot(
        t_long[pred_mask],
        x_long[pred_mask],
        label="Truth",
        color="#1d232f",
        lw=1.1,
    )
    axes[1].plot(
        t_long[pred_mask],
        prediction[pred_mask],
        label="HAVOK prediction",
        color="#2a9d8f",
        lw=1.1,
        alpha=0.95,
    )
    axes[1].set_title("Prediction using forcing from a longer run")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("x(t)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "reconstruction_prediction.png", bbox_inches="tight")
    plt.close(fig)


def save_forcing_alignment_figure(
    t: np.ndarray,
    x: np.ndarray,
    burst_times: np.ndarray,
    forcing: np.ndarray,
    threshold: float,
) -> None:
    forcing_time = t[: len(forcing)]
    window_start = max(0.0, float(burst_times[8]) - 5.0)
    window_end = min(float(t[-1]), float(burst_times[11]) + 5.0)

    signal_mask = get_time_mask(t, window_start, window_end)
    forcing_mask = get_time_mask(forcing_time, window_start, window_end)
    burst_window = burst_times[(burst_times >= window_start) & (burst_times <= window_end)]
    active_mask = np.abs(forcing[forcing_mask]) >= threshold

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t[signal_mask], x[signal_mask], color="#1d232f", lw=1.0)
    for burst_time in burst_window:
        axes[0].axvline(burst_time, color="#d1495b", lw=0.9, alpha=0.8)
    axes[0].set_title("Burst onsets detected from the scalar measurement")
    axes[0].set_ylabel("x(t)")

    axes[1].plot(forcing_time[forcing_mask], forcing[forcing_mask], color="#0b6e99", lw=1.0)
    axes[1].axhline(threshold, color="#d1495b", ls="--", lw=1.0)
    axes[1].axhline(-threshold, color="#d1495b", ls="--", lw=1.0)
    axes[1].scatter(
        forcing_time[forcing_mask][active_mask],
        forcing[forcing_mask][active_mask],
        color="#d1495b",
        s=14,
        zorder=3,
        label="|forcing| >= threshold",
    )
    for burst_time in burst_window:
        axes[1].axvline(burst_time, color="#f4a261", lw=0.9, alpha=0.5)
    axes[1].set_title("Sparse forcing concentrates near burst transitions")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Forcing")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "forcing_alignment.png", bbox_inches="tight")
    plt.close(fig)


def save_operator_spectrum_figure(operator: np.ndarray, singular_vals: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    vmax = float(np.abs(operator).max())
    image = axes[0].imshow(operator.real, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("HAVOK operator")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].semilogy(singular_vals, "o", color="#0b6e99", ms=4)
    axes[1].semilogy(singular_vals[:SVD_RANK], "o", color="#d1495b", ms=5)
    axes[1].axvline(SVD_RANK - 0.5, color="#f4a261", ls="--", lw=1.0)
    axes[1].set_title("Hankel singular values")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Singular value")

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "operator_spectrum.png", bbox_inches="tight")
    plt.close(fig)


def build_metrics(
    x: np.ndarray,
    reconstructed: np.ndarray,
    x_long: np.ndarray,
    prediction: np.ndarray,
    forcing: np.ndarray,
    threshold: float,
    t: np.ndarray,
    burst_indices: np.ndarray,
    operator: np.ndarray,
    singular_vals: np.ndarray,
) -> dict[str, float | int]:
    forcing_time = t[: len(forcing)]
    active_times = forcing_time[np.abs(forcing) >= threshold]
    burst_times = t[burst_indices]
    nearest_active = np.array([np.min(np.abs(active_times - bt)) for bt in burst_times])
    nearest_signed = np.array(
        [(active_times - bt)[np.argmin(np.abs(active_times - bt))] for bt in burst_times]
    )

    tridiag_mask = (
        np.abs(np.subtract.outer(np.arange(operator.shape[0]), np.arange(operator.shape[1])))
        <= 1
    )

    return {
        "dt": DT,
        "samples": M,
        "svd_rank": SVD_RANK,
        "delays": DELAYS,
        "reconstruction_rmse": float(np.sqrt(np.mean((x - reconstructed) ** 2))),
        "reconstruction_relative_rmse": float(
            np.sqrt(np.mean((x - reconstructed) ** 2)) / np.std(x)
        ),
        "reconstruction_correlation": float(np.corrcoef(x, reconstructed)[0, 1]),
        "prediction_rmse": float(np.sqrt(np.mean((x_long - prediction) ** 2))),
        "prediction_relative_rmse": float(
            np.sqrt(np.mean((x_long - prediction) ** 2)) / np.std(x_long)
        ),
        "prediction_correlation": float(np.corrcoef(x_long, prediction)[0, 1]),
        "forcing_threshold": float(threshold),
        "forcing_active_fraction": float(np.mean(np.abs(forcing) >= threshold)),
        "burst_count": int(len(burst_indices)),
        "max_burst_to_active_forcing_gap_s": float(np.max(nearest_active)),
        "median_burst_to_active_forcing_gap_s": float(np.median(nearest_active)),
        "mean_signed_burst_to_active_forcing_gap_s": float(np.mean(nearest_signed)),
        "median_signed_burst_to_active_forcing_gap_s": float(np.median(nearest_signed)),
        "fraction_bursts_with_nearest_active_forcing_before_onset": float(
            np.mean(nearest_signed <= 0)
        ),
        "rank5_energy_fraction": float(np.sum(singular_vals[:5] ** 2) / np.sum(singular_vals ** 2)),
        "rank11_energy_fraction": float(
            np.sum(singular_vals[:SVD_RANK] ** 2) / np.sum(singular_vals ** 2)
        ),
        "operator_skewness_residual": float(
            np.linalg.norm(operator + operator.T) / np.linalg.norm(operator)
        ),
        "operator_off_tridiagonal_fraction": float(
            np.linalg.norm(operator[~tridiag_mask]) / np.linalg.norm(operator)
        ),
    }


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    t = np.arange(M) * DT
    X = generate_hr_data(t)
    x = X[0]
    burst_indices = get_ind_burst_hr(x)

    havok = HAVOK(svd_rank=SVD_RANK, delays=DELAYS).fit(x, t)
    reconstructed = havok.reconstructed_data
    threshold = float(
        havok.compute_threshold(p=THRESHOLD_PROBABILITY, bins=100, plot=False)
    )
    forcing = havok.forcing[:, 0]

    t_long = np.arange(LONG_MULTIPLIER * M) * DT
    x_long = generate_hr_data(t_long)[0]
    havok_long = HAVOK(svd_rank=SVD_RANK, delays=DELAYS).fit(x_long, t_long)
    forcing_long = havok_long.forcing
    time_long = t_long[: len(forcing_long)]
    prediction = havok.predict(forcing_long, time_long)

    havok_full = HAVOK(svd_rank=-1, delays=DELAYS).fit(x, t)
    operator = havok.operator.real
    singular_vals = havok_full.singular_vals

    save_overview_figure(X, t, x)
    save_reconstruction_prediction_figure(
        t, x, reconstructed, t[burst_indices], t_long, x_long, prediction
    )
    save_forcing_alignment_figure(t, x, t[burst_indices], forcing, threshold)
    save_operator_spectrum_figure(operator, singular_vals)

    metrics = build_metrics(
        x,
        reconstructed,
        x_long,
        prediction,
        forcing,
        threshold,
        t,
        burst_indices,
        operator,
        singular_vals,
    )

    metrics_path = ASSET_DIR / "preliminary_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()