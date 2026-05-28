"""Standalone mHAVOK analysis for the Hindmarsh-Rose bursting regime.

This script mirrors the reproducible Lorenz mHAVOK workflow for the
Hindmarsh-Rose neuron model. It evaluates whether multichannel delay
embeddings improve burst-onset warning quality, then exports compact figures
and CSV summaries under plots/mhavok_hr.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


def hindmarsh_rose(
    _t: float,
    state: np.ndarray,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 1.0,
    d: float = 5.0,
    r: float = 0.0021,
    s: float = 4.0,
    x_rest: float = -1.6,
    I_ext: float = 3.281,
) -> list[float]:
    x_value, y_value, z_value = state
    dxdt = y_value - a * x_value**3 + b * x_value**2 - z_value + I_ext
    dydt = c - d * x_value**2 - y_value
    dzdt = r * (s * (x_value - x_rest) - z_value)
    return [dxdt, dydt, dzdt]


def generate_hr_data(
    t_eval: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    **kwargs: float,
) -> np.ndarray:
    solution = solve_ivp(
        lambda t_value, state: hindmarsh_rose(t_value, state, **kwargs),
        [float(t_eval[0]), float(t_eval[-1])],
        [0.0, 0.0, 0.0],
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution.y


def get_burst_indices(
    x_signal: np.ndarray,
    dt: float,
    height: float = 0.0,
    min_spike_gap: float = 5.0,
    min_burst_gap: float = 50.0,
) -> np.ndarray:
    spike_idx, _ = find_peaks(
        x_signal,
        height=height,
        distance=int(min_spike_gap / dt),
    )

    if len(spike_idx) == 0:
        return np.array([], dtype=int)

    gap_samples = int(min_burst_gap / dt)
    first_spikes = [spike_idx[0]]
    for index in range(1, len(spike_idx)):
        if spike_idx[index] - spike_idx[index - 1] > gap_samples:
            first_spikes.append(spike_idx[index])

    burst_onsets = []
    for spike in first_spikes:
        search_start = max(0, spike - int(50 / dt))
        segment = x_signal[search_start:spike]
        above_segment = (segment > height).astype(int)
        crossings = np.where(np.diff(above_segment) == 1)[0]
        if len(crossings) > 0:
            burst_onsets.append(search_start + crossings[-1])
        else:
            burst_onsets.append(spike)

    return np.array(burst_onsets, dtype=int)


def build_hankel(signal: np.ndarray, delays: int) -> np.ndarray:
    return np.array(
        [signal[index : index + len(signal) - delays + 1] for index in range(delays)]
    )


def get_event_onset_indices(active_mask: np.ndarray) -> np.ndarray:
    transitions = np.diff(active_mask.astype(int))
    onset_indices = np.where(transitions == 1)[0] + 1
    if active_mask[0]:
        onset_indices = np.insert(onset_indices, 0, 0)
    return onset_indices


def get_merged_event_onset_indices(
    active_mask: np.ndarray,
    dt: float,
    min_event_gap: float,
) -> np.ndarray:
    transitions = np.diff(active_mask.astype(int))
    onset_indices = np.where(transitions == 1)[0] + 1
    offset_indices = np.where(transitions == -1)[0] + 1

    if active_mask[0]:
        onset_indices = np.insert(onset_indices, 0, 0)
    if active_mask[-1]:
        offset_indices = np.append(offset_indices, len(active_mask) - 1)

    if len(onset_indices) == 0:
        return onset_indices

    gap_samples = int(min_event_gap / dt)
    merged_onsets = [onset_indices[0]]
    for index in range(1, len(onset_indices)):
        gap = onset_indices[index] - offset_indices[index - 1]
        if gap > gap_samples:
            merged_onsets.append(onset_indices[index])

    return np.array(merged_onsets, dtype=int)


def match_warning_events(
    burst_times: np.ndarray,
    prediction_times: np.ndarray,
    warning_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    matched_bursts = np.zeros(len(burst_times), dtype=bool)
    matched_predictions = np.zeros(len(prediction_times), dtype=bool)

    burst_index = 0
    prediction_index = 0
    while burst_index < len(burst_times) and prediction_index < len(prediction_times):
        gap = burst_times[burst_index] - prediction_times[prediction_index]
        if 0.0 <= gap <= warning_window:
            matched_bursts[burst_index] = True
            matched_predictions[prediction_index] = True
            burst_index += 1
            prediction_index += 1
        elif gap < 0.0:
            burst_index += 1
        else:
            prediction_index += 1

    return matched_bursts, matched_predictions


def compute_temporal_warning_labels(
    time_vector: np.ndarray,
    burst_times: np.ndarray,
    warning_window: float,
) -> np.ndarray:
    labels = np.zeros_like(time_vector, dtype=bool)
    for burst_time in burst_times:
        labels |= (time_vector >= burst_time - warning_window) & (time_vector <= burst_time)
    return labels


def compute_component_r2_scores(
    target_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
) -> np.ndarray:
    component_scores = []
    for component_index in range(target_matrix.shape[1]):
        target = target_matrix[:, component_index]
        prediction = prediction_matrix[:, component_index]
        ss_res = np.sum((target - prediction) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        component_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return np.array(component_scores)


def chamfer_distance_1d(reference_times: np.ndarray, candidate_times: np.ndarray) -> float:
    if len(reference_times) == 0 or len(candidate_times) == 0:
        return float("inf")
    pairwise_distances = np.abs(
        np.asarray(reference_times)[:, None] - np.asarray(candidate_times)[None, :]
    )
    return float(
        0.5
        * (
            np.mean(np.min(pairwise_distances, axis=1))
            + np.mean(np.min(pairwise_distances, axis=0))
        )
    )


def compute_nearest_signed_gaps(
    burst_times: np.ndarray,
    prediction_times: np.ndarray,
) -> np.ndarray:
    if len(prediction_times) == 0:
        return np.full(len(burst_times), np.inf, dtype=float)

    signed_gaps = []
    for burst_time in burst_times:
        time_diffs = prediction_times - burst_time
        signed_gaps.append(time_diffs[np.argmin(np.abs(time_diffs))])
    return np.array(signed_gaps, dtype=float)


def evaluate_mhavok_configuration(
    Y: np.ndarray,
    t: np.ndarray,
    dt: float,
    burst_times: np.ndarray,
    delays: int,
    rank: int,
    forcing_quantile: float = 0.80,
    warning_window: float = 20.0,
    min_event_gap: float = 30.0,
    return_series: bool = False,
) -> dict[str, object]:
    H = np.vstack([build_hankel(channel, delays) for channel in Y])
    _, singular_values, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh[:rank, :].T

    time_havok = t[delays - 1 : delays - 1 + V.shape[0]]
    valid_burst_times = burst_times[
        (burst_times >= time_havok[0]) & (burst_times <= time_havok[-1])
    ]

    forcing = V[:, rank - 1]
    V_linear = V[:, : rank - 1]
    dVdt = np.gradient(V_linear, dt, axis=0)
    Theta = np.column_stack([V_linear, forcing])

    model = LinearRegression(fit_intercept=False)
    model.fit(Theta, dVdt)
    Xi = model.coef_.T
    dVdt_pred = Theta @ Xi
    component_r2 = compute_component_r2_scores(dVdt, dVdt_pred)

    forcing_threshold = np.quantile(np.abs(forcing), forcing_quantile)
    active_forcing = np.abs(forcing) >= forcing_threshold
    event_onset_indices = get_merged_event_onset_indices(
        active_forcing,
        dt,
        min_event_gap,
    )
    event_times = time_havok[event_onset_indices]

    nearest_signed_gaps = compute_nearest_signed_gaps(valid_burst_times, event_times)
    matched_bursts, matched_predictions = match_warning_events(
        valid_burst_times,
        event_times,
        warning_window,
    )

    true_positive = int(matched_bursts.sum())
    false_negative = int(len(valid_burst_times) - true_positive)
    false_positive = int(len(event_times) - true_positive)

    event_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    event_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    event_f1 = 0.0
    if event_precision + event_recall > 0:
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall)

    true_labels = compute_temporal_warning_labels(time_havok, valid_burst_times, warning_window)
    pred_labels = active_forcing
    event_accuracy = float(np.mean(true_labels == pred_labels))
    chamfer_distance = chamfer_distance_1d(valid_burst_times, event_times)

    matched_lead_times = valid_burst_times[matched_bursts] - event_times[matched_predictions]
    warning_hits = (-warning_window <= nearest_signed_gaps) & (nearest_signed_gaps <= 0.0)

    result: dict[str, object] = {
        "delays": delays,
        "rank": rank,
        "warning_window": warning_window,
        "min_event_gap": min_event_gap,
        "median_abs_gap": float(np.median(np.abs(nearest_signed_gaps))),
        "median_signed_gap": float(np.median(nearest_signed_gaps)),
        "fraction_preceding": float(np.mean(nearest_signed_gaps <= 0.0)),
        "fraction_within_window": float(np.mean(warning_hits)),
        "event_recall": float(event_recall),
        "event_precision": float(event_precision),
        "event_accuracy": event_accuracy,
        "event_f1": float(event_f1),
        "chamfer_distance": float(chamfer_distance),
        "mean_linear_r2": float(np.mean(component_r2)),
        "min_linear_r2": float(np.min(component_r2)),
        "max_linear_r2": float(np.max(component_r2)),
        "mean_lead_time": float(np.mean(matched_lead_times)) if len(matched_lead_times) > 0 else np.nan,
        "median_lead_time": float(np.median(matched_lead_times)) if len(matched_lead_times) > 0 else np.nan,
        "active_fraction": float(np.mean(active_forcing)),
        "n_bursts": int(len(valid_burst_times)),
        "n_predictions": int(len(event_times)),
    }

    if return_series:
        result.update(
            {
                "forcing": forcing,
                "time_havok": time_havok,
                "active_forcing": active_forcing,
                "forcing_threshold": float(forcing_threshold),
                "nearest_signed_gaps": nearest_signed_gaps,
                "singular_values": singular_values,
                "augmented_operator": Xi.T,
                "component_r2": component_r2,
                "event_times": event_times,
                "true_labels": true_labels,
                "pred_labels": pred_labels,
            }
        )

    return result


def build_metric_grid(
    records: list[dict[str, object]],
    row_values: list[int],
    col_values: list[int],
    metric_key: str,
) -> np.ndarray:
    grid = np.full((len(row_values), len(col_values)), np.nan)
    for row_index, trial_delays in enumerate(row_values):
        for col_index, trial_rank in enumerate(col_values):
            for record in records:
                if record["delays"] == trial_delays and record["rank"] == trial_rank:
                    grid[row_index, col_index] = float(record[metric_key])
                    break
    return grid


def save_rank_delay_heatmaps(
    sweep_results: list[dict[str, object]],
    delay_grid: list[int],
    rank_grid: list[int],
    output_dir: Path,
) -> None:
    heatmap_specs = [
        ("event_recall", "Recall", "YlGn", 0.0, 1.0),
        ("event_precision", "Precision", "YlGn", 0.0, 1.0),
        ("event_f1", "Event F1", "YlGn", 0.0, 1.0),
        ("mean_linear_r2", "Mean linear-component R²", "cividis", 0.0, 1.0),
        ("fraction_within_window", "Fraction within warning window", "YlGn", 0.0, 1.0),
        ("median_abs_gap", "Median |gap| (tu)", "viridis_r", None, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (metric_key, title, cmap, vmin, vmax) in zip(axes.flat, heatmap_specs):
        grid = build_metric_grid(sweep_results, delay_grid, rank_grid, metric_key)
        image = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(rank_grid)))
        ax.set_xticklabels(rank_grid)
        ax.set_yticks(range(len(delay_grid)))
        ax.set_yticklabels(delay_grid)
        ax.set_xlabel("rank")
        ax.set_ylabel("delays")
        ax.set_title(title)

        finite_values = grid[np.isfinite(grid)]
        annotation_threshold = np.nanmedian(finite_values) if finite_values.size > 0 else 0.0
        for row_index, _ in enumerate(delay_grid):
            for col_index, _ in enumerate(rank_grid):
                value = grid[row_index, col_index]
                label = "nan" if np.isnan(value) else f"{value:.3f}"
                text_color = (
                    "white"
                    if np.isfinite(value) and value <= annotation_threshold
                    else "black"
                )
                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Hindmarsh-Rose mHAVOK sensitivity to rank and delays", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "mhavok_hr_rank_delay_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    metric_specs = [
        ("event recall", "Recall"),
        ("event precision", "Precision"),
        ("event F1", "Event F1"),
        ("mean linear R²", "Mean linear-component R²"),
        ("mean lead time (tu)", "Mean matched lead (tu)"),
        ("median |gap| (tu)", "Median |gap| (tu)"),
    ]
    positions = np.arange(len(comparison_df))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (metric_key, title) in zip(axes.flat, metric_specs):
        values = comparison_df[metric_key].to_numpy(dtype=float)
        ax.bar(positions, values, color=colors)
        ax.set_xticks(positions)
        ax.set_xticklabels(comparison_df["model"], rotation=18, ha="right")
        ax.set_title(title)
        finite_values = values[np.isfinite(values)]
        upper = finite_values.max() if finite_values.size > 0 else 1.0
        lower = finite_values.min() if finite_values.size > 0 else 0.0
        if metric_key in {"event recall", "event precision", "event F1", "mean linear R²"}:
            ax.set_ylim(0, max(1.0, 1.08 * upper))
            offset = 0.02
        else:
            ax.set_ylim(min(0.0, 1.08 * lower), 1.12 * upper if upper > 0 else 1.0)
            offset = max(0.02 * max(abs(upper), abs(lower), 1.0), 0.05)
        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.text(index, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Hindmarsh-Rose x-only baseline vs tuned mHAVOK models", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "mhavok_hr_x_only_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_channel_combo_plot(combo_best_df: pd.DataFrame, output_dir: Path) -> None:
    metric_specs = [
        ("event recall", "Recall"),
        ("event precision", "Precision"),
        ("event F1", "Event F1"),
        ("mean linear R²", "Mean linear-component R²"),
        ("mean lead time (tu)", "Mean matched lead (tu)"),
        ("median |gap| (tu)", "Median |gap| (tu)"),
    ]
    positions = np.arange(len(combo_best_df))
    colors = plt.cm.tab20(np.linspace(0.05, 0.95, len(combo_best_df)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric_key, title) in zip(axes.flat, metric_specs):
        values = combo_best_df[metric_key].to_numpy(dtype=float)
        ax.bar(positions, values, color=colors)
        ax.set_xticks(positions)
        ax.set_xticklabels(combo_best_df["channel combo"], rotation=45, ha="right")
        ax.set_title(title)
        finite_values = values[np.isfinite(values)]
        upper = finite_values.max() if finite_values.size > 0 else 1.0
        lower = finite_values.min() if finite_values.size > 0 else 0.0
        if metric_key in {"event recall", "event precision", "event F1", "mean linear R²"}:
            ax.set_ylim(0, max(1.0, 1.08 * upper))
            offset = 0.02
        else:
            ax.set_ylim(min(0.0, 1.08 * lower), 1.12 * upper if upper > 0 else 1.0)
            offset = max(0.02 * max(abs(upper), abs(lower), 1.0), 0.05)
        for index, row in combo_best_df.iterrows():
            value = values[index]
            if np.isfinite(value):
                ax.text(
                    index,
                    value + offset,
                    f"d={row['delays']}, r={row['rank']}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    plt.suptitle("Best Hindmarsh-Rose mHAVOK metrics by observable combination", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "mhavok_hr_channel_combo_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_component_r2_plot(component_r2_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    values = component_r2_df["R²"].to_numpy(dtype=float)
    positions = np.arange(len(component_r2_df))
    ax.bar(positions, values, color="tab:blue")
    ax.set_xticks(positions)
    ax.set_xticklabels(component_r2_df["component"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²")
    ax.set_title("Tuned Hindmarsh-Rose mHAVOK component R²")
    for index, value in enumerate(values):
        ax.text(index, min(value + 0.02, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / "mhavok_hr_component_r2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_results(
    output_dir: Path,
    metadata_df: pd.DataFrame,
    sweep_metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    combo_best_df: pd.DataFrame,
    component_r2_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_comparison_path = output_dir / "x_only_baseline_vs_tuned.csv"
    if legacy_comparison_path.exists():
        legacy_comparison_path.unlink()
    metadata_df.round(4).to_csv(output_dir / "analysis_metadata.csv", index=False)
    sweep_metrics_df.round(4).to_csv(output_dir / "rank_delay_event_metrics.csv", index=False)
    comparison_df.round(4).to_csv(output_dir / "x_only_vs_tuned.csv", index=False)
    combo_best_df.round(4).to_csv(output_dir / "all_channel_combo_metrics.csv", index=False)
    component_r2_df.round(4).to_csv(output_dir / "tuned_component_r2.csv", index=False)


def run_analysis(quick: bool = False, analysis_points: int | None = None) -> None:
    dt = 0.1
    transient_time = 950.0
    warning_window = 20.0
    forcing_quantile = 0.80
    min_event_gap = 30.0
    delay_grid = [50, 100] if quick else [50, 100, 150]
    rank_grid = [5, 7, 9] if quick else [5, 7, 9, 11, 13]
    default_analysis_points = 30_000 if quick else 80_000
    analysis_points = default_analysis_points if analysis_points is None else analysis_points

    transient_points = int(transient_time / dt)
    total_points = transient_points + analysis_points
    t_full = np.arange(total_points) * dt
    X_full = generate_hr_data(t_full)

    X = X_full[:, transient_points:]
    t = t_full[transient_points:] - t_full[transient_points]
    burst_indices = get_burst_indices(X[0], dt)
    burst_times = t[burst_indices]

    if len(burst_times) < 5:
        raise RuntimeError("Not enough HR bursts detected for mHAVOK evaluation.")

    print("System summary:")
    print(f"  analysis points = {analysis_points}")
    print(f"  burst onsets    = {len(burst_times)}")
    print(f"  warning window  = {warning_window:.1f} time units")
    print(f"  forcing quantile= {forcing_quantile:.2f}")
    print(f"  merge gap       = {min_event_gap:.1f} time units")
    print()

    full_combo = X[[0, 1, 2], :]
    baseline_combo = X[[0], :]

    sweep_results = []
    for trial_delays in delay_grid:
        for trial_rank in rank_grid:
            sweep_results.append(
                evaluate_mhavok_configuration(
                    full_combo,
                    t,
                    dt,
                    burst_times,
                    trial_delays,
                    trial_rank,
                    forcing_quantile=forcing_quantile,
                    warning_window=warning_window,
                    min_event_gap=min_event_gap,
                )
            )

    sorted_by_alignment = sorted(
        sweep_results,
        key=lambda row: (row["median_abs_gap"], -row["fraction_within_window"], -row["event_recall"]),
    )
    sorted_by_event = sorted(
        sweep_results,
        key=lambda row: (
            -row["event_f1"],
            -row["event_recall"],
            -row["event_precision"],
            row["median_abs_gap"],
        ),
    )

    best_alignment_setting = sorted_by_alignment[0]
    best_event_setting = sorted_by_event[0]
    best_alignment_result = evaluate_mhavok_configuration(
        full_combo,
        t,
        dt,
        burst_times,
        int(best_alignment_setting["delays"]),
        int(best_alignment_setting["rank"]),
        forcing_quantile=forcing_quantile,
        warning_window=warning_window,
        min_event_gap=min_event_gap,
        return_series=True,
    )
    best_event_result = evaluate_mhavok_configuration(
        full_combo,
        t,
        dt,
        burst_times,
        int(best_event_setting["delays"]),
        int(best_event_setting["rank"]),
        forcing_quantile=forcing_quantile,
        warning_window=warning_window,
        min_event_gap=min_event_gap,
        return_series=True,
    )

    combo_sets = {
        "y": X[[1], :],
        "z": X[[2], :],
        "x": baseline_combo,
        "x+y": X[[0, 1], :],
        "x+z": X[[0, 2], :],
        "y+z": X[[1, 2], :],
        "x+y+z": X[[0, 1, 2], :],
    }
    combo_best_rows = []
    for combo_name, combo_data in combo_sets.items():
        combo_results = []
        for trial_delays in delay_grid:
            for trial_rank in rank_grid:
                result = evaluate_mhavok_configuration(
                    combo_data,
                    t,
                    dt,
                    burst_times,
                    trial_delays,
                    trial_rank,
                    forcing_quantile=forcing_quantile,
                    warning_window=warning_window,
                    min_event_gap=min_event_gap,
                )
                result["channel combo"] = combo_name
                combo_results.append(result)

        best_combo_result = max(
            combo_results,
            key=lambda row: (
                row["event_f1"],
                row["event_recall"],
                row["event_precision"],
                row["mean_linear_r2"],
                -row["median_abs_gap"],
            ),
        )
        combo_best_rows.append(
            {
                "channel combo": combo_name,
                "delays": best_combo_result["delays"],
                "rank": best_combo_result["rank"],
                "event recall": best_combo_result["event_recall"],
                "event precision": best_combo_result["event_precision"],
                "event F1": best_combo_result["event_f1"],
                "mean linear R²": best_combo_result["mean_linear_r2"],
                "mean lead time (tu)": best_combo_result["mean_lead_time"],
                "median |gap| (tu)": best_combo_result["median_abs_gap"],
            }
        )

    combo_best_df = pd.DataFrame(combo_best_rows).sort_values(
        [
            "event F1",
            "event recall",
            "event precision",
            "mean linear R²",
            "median |gap| (tu)",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    best_x_only_row = combo_best_df.loc[combo_best_df["channel combo"] == "x"].iloc[0]
    comparison_df = pd.DataFrame(
        [
            {
                "model": "best x only",
                "observable set": "x",
                "delays": best_x_only_row["delays"],
                "rank": best_x_only_row["rank"],
                "event recall": best_x_only_row["event recall"],
                "event precision": best_x_only_row["event precision"],
                "event F1": best_x_only_row["event F1"],
                "mean linear R²": best_x_only_row["mean linear R²"],
                "mean lead time (tu)": best_x_only_row["mean lead time (tu)"],
                "median |gap| (tu)": best_x_only_row["median |gap| (tu)"],
            },
            {
                "model": "best x+y+z alignment",
                "observable set": "x+y+z",
                "delays": best_alignment_result["delays"],
                "rank": best_alignment_result["rank"],
                "event recall": best_alignment_result["event_recall"],
                "event precision": best_alignment_result["event_precision"],
                "event F1": best_alignment_result["event_f1"],
                "mean linear R²": best_alignment_result["mean_linear_r2"],
                "mean lead time (tu)": best_alignment_result["mean_lead_time"],
                "median |gap| (tu)": best_alignment_result["median_abs_gap"],
            },
            {
                "model": "best x+y+z warning",
                "observable set": "x+y+z",
                "delays": best_event_result["delays"],
                "rank": best_event_result["rank"],
                "event recall": best_event_result["event_recall"],
                "event precision": best_event_result["event_precision"],
                "event F1": best_event_result["event_f1"],
                "mean linear R²": best_event_result["mean_linear_r2"],
                "mean lead time (tu)": best_event_result["mean_lead_time"],
                "median |gap| (tu)": best_event_result["median_abs_gap"],
            },
        ]
    )

    component_r2_df = pd.DataFrame(
        {
            "component": [
                f"v{index + 1}" for index in range(len(best_event_result["component_r2"]))
            ],
            "R²": best_event_result["component_r2"],
        }
    )

    sweep_metrics_df = pd.DataFrame(sweep_results).sort_values(
        ["event_f1", "event_recall", "event_precision", "mean_linear_r2"],
        ascending=[False, False, False, False],
    )
    metadata_df = pd.DataFrame(
        [
            {
                "dt": dt,
                "analysis_points": analysis_points,
                "analysis_duration": t[-1],
                "burst_onsets": len(burst_times),
                "warning_window": warning_window,
                "forcing_quantile": forcing_quantile,
                "min_event_gap": min_event_gap,
            }
        ]
    )

    output_dir = Path("plots") / "mhavok_hr"
    export_results(
        output_dir,
        metadata_df,
        sweep_metrics_df,
        comparison_df,
        combo_best_df,
        component_r2_df,
    )
    save_rank_delay_heatmaps(sweep_metrics_df.to_dict("records"), delay_grid, rank_grid, output_dir)
    save_model_comparison_plot(comparison_df, output_dir)
    save_channel_combo_plot(combo_best_df, output_dir)
    save_component_r2_plot(component_r2_df, output_dir)

    print("Best x-only configuration:")
    print(
        f"  delays={int(best_x_only_row['delays'])}, rank={int(best_x_only_row['rank'])}, "
        f"recall={best_x_only_row['event recall']:.3f}, "
        f"precision={best_x_only_row['event precision']:.3f}, "
        f"F1={best_x_only_row['event F1']:.3f}"
    )
    print()
    print("Best x+y+z setting by alignment:")
    print(
        f"  delays={best_alignment_setting['delays']}, rank={best_alignment_setting['rank']}, "
        f"median |gap|={best_alignment_setting['median_abs_gap']:.3f} tu, "
        f"recall={best_alignment_setting['event_recall']:.3f}, "
        f"precision={best_alignment_setting['event_precision']:.3f}"
    )
    print()
    print("Best x+y+z setting by warning F1:")
    print(
        f"  delays={best_event_setting['delays']}, rank={best_event_setting['rank']}, "
        f"recall={best_event_setting['event_recall']:.3f}, "
        f"precision={best_event_setting['event_precision']:.3f}, "
        f"F1={best_event_setting['event_f1']:.3f}, "
        f"mean lead={best_event_setting['mean_lead_time']:.3f} tu"
    )
    print()
    print("Best observable combination by warning F1:")
    print(combo_best_df.round(4).iloc[0].to_string())
    print()
    print("Exported CSV summaries and plots to plots/mhavok_hr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the standalone Hindmarsh-Rose mHAVOK analysis."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a shorter trajectory and smaller sweep for a fast smoke test.",
    )
    parser.add_argument(
        "--analysis-points",
        type=int,
        default=None,
        help="Override the number of post-transient samples used for analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(quick=args.quick, analysis_points=args.analysis_points)


if __name__ == "__main__":
    main()