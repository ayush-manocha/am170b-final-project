"""Manual mHAVOK implementation for the Lorenz system.

This script extracts the core implementation from mHAVOK_lorenz.ipynb into a
standalone, reproducible Python entry point. It reproduces the notebook's main
rank-delay sweep, observable-set comparison, and all-channel benchmark, then
exports the same CSV summaries under plots/mhavok_lorenz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression


def generate_lorenz_data(t_eval: np.ndarray) -> np.ndarray:
    def lorenz_system(_t: float, state: np.ndarray) -> list[float]:
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        x_value, y_value, z_value = state
        dxdt = sigma * (y_value - x_value)
        dydt = x_value * (rho - z_value) - y_value
        dzdt = x_value * y_value - beta * z_value
        return [dxdt, dydt, dzdt]

    solution = solve_ivp(
        lorenz_system,
        [float(t_eval[0]), float(t_eval[-1])],
        [-8.0, 8.0, 27.0],
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-12,
        atol=1e-12,
    )
    return solution.y


def build_hankel(signal: np.ndarray, delays: int) -> np.ndarray:
    return np.array(
        [signal[index : index + len(signal) - delays + 1] for index in range(delays)]
    )


def get_lorenz_switch_indices(x_signal: np.ndarray) -> np.ndarray:
    switch_mask = np.sign(x_signal[:-1]) - np.sign(x_signal[1:]) != 0
    switch_mask = np.append(switch_mask, False)
    return np.where(switch_mask)[0]


def get_event_onset_indices(active_mask: np.ndarray) -> np.ndarray:
    transitions = np.diff(active_mask.astype(int))
    onset_indices = np.where(transitions == 1)[0] + 1
    if active_mask[0]:
        onset_indices = np.insert(onset_indices, 0, 0)
    return onset_indices


def match_event_times(
    reference_times: np.ndarray,
    candidate_times: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference_times = np.asarray(reference_times)
    candidate_times = np.asarray(candidate_times)
    matched_reference = np.zeros(len(reference_times), dtype=bool)
    matched_candidate = np.zeros(len(candidate_times), dtype=bool)

    reference_index = 0
    candidate_index = 0
    while reference_index < len(reference_times) and candidate_index < len(candidate_times):
        time_gap = candidate_times[candidate_index] - reference_times[reference_index]
        if abs(time_gap) <= tolerance:
            matched_reference[reference_index] = True
            matched_candidate[candidate_index] = True
            reference_index += 1
            candidate_index += 1
        elif time_gap < -tolerance:
            candidate_index += 1
        else:
            reference_index += 1

    return matched_reference, matched_candidate


def compute_temporal_labels(
    time_vector: np.ndarray,
    event_times: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    labels = np.zeros_like(time_vector, dtype=bool)
    for event_time in event_times:
        labels |= np.abs(time_vector - event_time) <= tolerance
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


def evaluate_mhavok_configuration(
    Y: np.ndarray,
    t: np.ndarray,
    dt: float,
    switch_times: np.ndarray,
    delays: int,
    rank: int,
    forcing_quantile: float = 0.95,
    event_tolerance: float = 0.10,
    return_series: bool = False,
) -> dict[str, object]:
    H = np.vstack([build_hankel(channel, delays) for channel in Y])
    _, singular_values, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh[:rank, :].T
    time_havok = t[delays - 1 : delays - 1 + V.shape[0]]
    forcing = V[:, rank - 1]
    V_linear = V[:, : rank - 1]

    dVdt = np.gradient(V_linear, dt, axis=0)
    Theta = np.column_stack([V_linear, forcing])
    model = LinearRegression(fit_intercept=False)
    model.fit(Theta, dVdt)

    Xi = model.coef_.T
    A = Xi[: rank - 1, :]
    B = Xi[rank - 1 :, :]
    dVdt_pred = Theta @ Xi
    component_r2 = compute_component_r2_scores(dVdt, dVdt_pred)

    forcing_threshold = np.quantile(np.abs(forcing), forcing_quantile)
    active_forcing = np.abs(forcing) >= forcing_threshold
    active_times = time_havok[active_forcing]
    event_onset_indices = get_event_onset_indices(active_forcing)
    event_times = time_havok[event_onset_indices]

    nearest_signed_gaps = []
    for switch_time in switch_times:
        time_diffs = active_times - switch_time
        nearest_signed_gaps.append(time_diffs[np.argmin(np.abs(time_diffs))])
    nearest_signed_gaps = np.array(nearest_signed_gaps)

    matched_switches, _ = match_event_times(
        switch_times,
        event_times,
        event_tolerance,
    )
    true_positive = int(matched_switches.sum())
    false_negative = int(len(switch_times) - true_positive)
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

    true_labels = compute_temporal_labels(time_havok, switch_times, event_tolerance)
    pred_labels = active_forcing
    event_accuracy = float(np.mean(true_labels == pred_labels))
    chamfer_distance = chamfer_distance_1d(switch_times, event_times)

    result: dict[str, object] = {
        "delays": delays,
        "rank": rank,
        "median_abs_gap": float(np.median(np.abs(nearest_signed_gaps))),
        "median_signed_gap": float(np.median(nearest_signed_gaps)),
        "fraction_preceding": float(np.mean(nearest_signed_gaps <= 0)),
        "fraction_within_0p10": float(np.mean(np.abs(nearest_signed_gaps) <= 0.10)),
        "fraction_within_0p25": float(np.mean(np.abs(nearest_signed_gaps) <= 0.25)),
        "event_recall": float(event_recall),
        "event_precision": float(event_precision),
        "event_accuracy": event_accuracy,
        "event_f1": float(event_f1),
        "chamfer_distance": float(chamfer_distance),
        "mean_linear_r2": float(np.mean(component_r2)),
        "min_linear_r2": float(np.min(component_r2)),
        "max_linear_r2": float(np.max(component_r2)),
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
                "A": A,
                "B": B,
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
        ("event_accuracy", "Accuracy", "YlGn", 0.0, 1.0),
        ("event_f1", "Event F1", "YlGn", 0.0, 1.0),
        ("mean_linear_r2", "Mean linear-component R²", "cividis", 0.0, 1.0),
        ("chamfer_distance", "Chamfer distance (s)", "viridis_r", None, None),
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
        if finite_values.size > 0:
            annotation_threshold = np.nanmedian(finite_values)
        else:
            annotation_threshold = 0.0
        for row_index, trial_delays in enumerate(delay_grid):
            for col_index, trial_rank in enumerate(rank_grid):
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

    plt.suptitle("mHAVOK sensitivity to rank and delays", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "mhavok_lorenz_rank_delay_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_xz_model_comparison_plot(
    baseline_result: dict[str, object],
    best_alignment_result: dict[str, object],
    best_event_result: dict[str, object],
    output_dir: Path,
) -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "model": "baseline x+z",
                "event recall": baseline_result["event_recall"],
                "event precision": baseline_result["event_precision"],
                "event accuracy": baseline_result["event_accuracy"],
                "event F1": baseline_result["event_f1"],
                "mean linear R²": baseline_result["mean_linear_r2"],
                "Chamfer distance (s)": baseline_result["chamfer_distance"],
            },
            {
                "model": "best alignment",
                "event recall": best_alignment_result["event_recall"],
                "event precision": best_alignment_result["event_precision"],
                "event accuracy": best_alignment_result["event_accuracy"],
                "event F1": best_alignment_result["event_f1"],
                "mean linear R²": best_alignment_result["mean_linear_r2"],
                "Chamfer distance (s)": best_alignment_result["chamfer_distance"],
            },
            {
                "model": "best event",
                "event recall": best_event_result["event_recall"],
                "event precision": best_event_result["event_precision"],
                "event accuracy": best_event_result["event_accuracy"],
                "event F1": best_event_result["event_f1"],
                "mean linear R²": best_event_result["mean_linear_r2"],
                "Chamfer distance (s)": best_event_result["chamfer_distance"],
            },
        ]
    )

    metric_specs = [
        ("event recall", "Recall"),
        ("event precision", "Precision"),
        ("event accuracy", "Accuracy"),
        ("event F1", "Event F1"),
        ("mean linear R²", "Mean linear-component R²"),
        ("Chamfer distance (s)", "Chamfer distance (s)"),
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
        if metric_key != "Chamfer distance (s)":
            ax.set_ylim(0, max(1.0, 1.08 * values.max()))
            offset = 0.02
        else:
            offset = max(0.01 * values.max(), 1e-3)
        for index, value in enumerate(values):
            ax.text(index, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("mHAVOK x+z configuration comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "mhavok_lorenz_xz_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_channel_combo_plot(combo_best_df: pd.DataFrame, output_dir: Path) -> None:
    metric_specs = [
        ("event recall", "Recall"),
        ("event precision", "Precision"),
        ("event accuracy", "Accuracy"),
        ("event F1", "Event F1"),
        ("mean linear R²", "Mean linear-component R²"),
        ("Chamfer distance (s)", "Chamfer distance (s)"),
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
        if metric_key != "Chamfer distance (s)":
            ax.set_ylim(0, max(1.0, 1.08 * values.max()))
            offset = 0.02
        else:
            offset = max(0.01 * values.max(), 1e-3)
        for index, row in combo_best_df.iterrows():
            ax.text(
                index,
                values[index] + offset,
                f"d={row['delays']}, r={row['rank']}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.suptitle("Best mHAVOK metrics by observable combination", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "mhavok_lorenz_channel_combo_metrics.png", dpi=200, bbox_inches="tight")
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
    ax.set_title("Tuned mHAVOK component R²")
    for index, value in enumerate(values):
        ax.text(index, min(value + 0.02, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / "mhavok_lorenz_component_r2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_grid(
    Y: np.ndarray,
    t: np.ndarray,
    dt: float,
    switch_times: np.ndarray,
    delay_grid: list[int],
    rank_grid: list[int],
    event_tolerance: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    sweep_results = []
    for trial_delays in delay_grid:
        for trial_rank in rank_grid:
            sweep_results.append(
                evaluate_mhavok_configuration(
                    Y,
                    t,
                    dt,
                    switch_times,
                    trial_delays,
                    trial_rank,
                    event_tolerance=event_tolerance,
                )
            )

    sorted_by_alignment = sorted(
        sweep_results,
        key=lambda row: (row["median_abs_gap"], -row["fraction_within_0p10"]),
    )
    sorted_by_event = sorted(
        sweep_results,
        key=lambda row: (
            -row["event_f1"],
            -row["event_recall"],
            -row["event_precision"],
            row["chamfer_distance"],
        ),
    )
    sweep_metrics_df = pd.DataFrame(sweep_results).sort_values(
        ["event_f1", "event_recall", "event_precision", "mean_linear_r2"],
        ascending=[False, False, False, False],
    )
    return sweep_results, sorted_by_alignment, sorted_by_event, sweep_metrics_df


def summarize_best_observable_models(
    X: np.ndarray,
    t: np.ndarray,
    dt: float,
    switch_times: np.ndarray,
    delay_grid: list[int],
    rank_grid: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def normalize(values: np.ndarray) -> np.ndarray:
        value_range = values.max() - values.min()
        if value_range == 0:
            return np.ones_like(values, dtype=float)
        return (values - values.min()) / value_range

    channel_sets = {
        "x only": X[[0], :],
        "z only": X[[2], :],
        "x and z": X[[0, 2], :],
    }

    channel_results = []
    for channel_name, channel_data in channel_sets.items():
        for trial_delays in delay_grid:
            for trial_rank in rank_grid:
                result = evaluate_mhavok_configuration(
                    channel_data,
                    t,
                    dt,
                    switch_times,
                    trial_delays,
                    trial_rank,
                )
                result["channel_name"] = channel_name
                channel_results.append(result)

    best_by_channel = []
    for channel_name in channel_sets:
        best_by_channel.append(
            min(
                [row for row in channel_results if row["channel_name"] == channel_name],
                key=lambda row: (row["median_abs_gap"], -row["fraction_within_0p10"]),
            )
        )

    summary_df = pd.DataFrame(
        [
            {
                "observable set": row["channel_name"],
                "delays": row["delays"],
                "rank": row["rank"],
                "median |gap| (s)": row["median_abs_gap"],
                "median signed gap (s)": row["median_signed_gap"],
                "fraction within 0.10 s": row["fraction_within_0p10"],
                "fraction preceding": row["fraction_preceding"],
            }
            for row in best_by_channel
        ]
    ).sort_values("median |gap| (s)")

    gap_summary_df = pd.DataFrame(
        [
            {
                "observable set": row["channel_name"],
                "median signed gap (s)": row["median_signed_gap"],
                "fraction preceding": row["fraction_preceding"],
            }
            for row in best_by_channel
        ]
    ).sort_values("median signed gap (s)")

    selection_df = summary_df.copy().reset_index(drop=True)
    median_gap_values = selection_df["median |gap| (s)"].to_numpy()
    fraction_within_values = selection_df["fraction within 0.10 s"].to_numpy()
    fraction_preceding_values = selection_df["fraction preceding"].to_numpy()

    selection_df["gap score"] = 1 - normalize(median_gap_values)
    selection_df["within-0.10 score"] = normalize(fraction_within_values)
    selection_df["preceding score"] = normalize(fraction_preceding_values)

    selection_profiles = {
        "tight alignment": {
            "gap score": 0.60,
            "within-0.10 score": 0.20,
            "preceding score": 0.20,
        },
        "balanced": {
            "gap score": 0.40,
            "within-0.10 score": 0.30,
            "preceding score": 0.30,
        },
        "early warning": {
            "gap score": 0.20,
            "within-0.10 score": 0.40,
            "preceding score": 0.40,
        },
    }

    for profile_name, weights in selection_profiles.items():
        composite_score = (
            weights["gap score"] * selection_df["gap score"]
            + weights["within-0.10 score"] * selection_df["within-0.10 score"]
            + weights["preceding score"] * selection_df["preceding score"]
        )
        selection_df[f"score: {profile_name}"] = composite_score

    return summary_df, gap_summary_df, selection_df


def summarize_all_channel_combos(
    X: np.ndarray,
    t: np.ndarray,
    dt: float,
    switch_times: np.ndarray,
    delay_grid: list[int],
    rank_grid: list[int],
    event_tolerance: float,
) -> pd.DataFrame:
    all_channel_sets = {
        "x": X[[0], :],
        "y": X[[1], :],
        "z": X[[2], :],
        "x+y": X[[0, 1], :],
        "x+z": X[[0, 2], :],
        "y+z": X[[1, 2], :],
        "x+y+z": X[[0, 1, 2], :],
    }

    combo_best_rows = []
    for combo_name, combo_data in all_channel_sets.items():
        combo_results = []
        for trial_delays in delay_grid:
            for trial_rank in rank_grid:
                result = evaluate_mhavok_configuration(
                    combo_data,
                    t,
                    dt,
                    switch_times,
                    trial_delays,
                    trial_rank,
                    event_tolerance=event_tolerance,
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
                -row["chamfer_distance"],
            ),
        )

        combo_best_rows.append(
            {
                "channel combo": combo_name,
                "delays": best_combo_result["delays"],
                "rank": best_combo_result["rank"],
                "event recall": best_combo_result["event_recall"],
                "event precision": best_combo_result["event_precision"],
                "event accuracy": best_combo_result["event_accuracy"],
                "event F1": best_combo_result["event_f1"],
                "mean linear R²": best_combo_result["mean_linear_r2"],
                "Chamfer distance (s)": best_combo_result["chamfer_distance"],
                "median |gap| (s)": best_combo_result["median_abs_gap"],
            }
        )

    return pd.DataFrame(combo_best_rows).sort_values(
        ["event F1", "event recall", "event precision", "mean linear R²", "Chamfer distance (s)"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def export_results(
    output_dir: Path,
    summary_df: pd.DataFrame,
    gap_summary_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    sweep_metrics_df: pd.DataFrame,
    combo_best_df: pd.DataFrame,
    component_r2_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.round(4).to_csv(output_dir / "best_observable_models.csv", index=False)
    gap_summary_df.round(4).to_csv(output_dir / "signed_gap_summary.csv", index=False)
    selection_df.round(4).to_csv(output_dir / "objective_model_selection.csv", index=False)
    sweep_metrics_df.round(4).to_csv(output_dir / "rank_delay_event_metrics.csv", index=False)
    combo_best_df.round(4).to_csv(output_dir / "all_channel_combo_metrics.csv", index=False)
    component_r2_df.round(4).to_csv(output_dir / "tuned_component_r2.csv", index=False)


def run_analysis(quick: bool = False) -> None:
    dt = 0.002 if quick else 0.001
    m = 10000 if quick else 50000
    delay_grid = [50, 100] if quick else [50, 100, 150]
    rank_grid = [5, 7, 9] if quick else [5, 7, 9, 11, 13]
    baseline_delays = 100
    baseline_rank = 9
    event_tolerance = 0.10

    t = np.arange(m) * dt
    X = generate_lorenz_data(t)
    Y = X[[0, 2], :]
    switch_indices = get_lorenz_switch_indices(X[0])
    switch_times = t[switch_indices]

    baseline_result = evaluate_mhavok_configuration(
        Y,
        t,
        dt,
        switch_times,
        baseline_delays,
        baseline_rank,
        return_series=True,
        event_tolerance=event_tolerance,
    )

    _, sorted_by_alignment, sorted_by_event, sweep_metrics_df = evaluate_grid(
        Y,
        t,
        dt,
        switch_times,
        delay_grid,
        rank_grid,
        event_tolerance,
    )

    best_alignment_setting = sorted_by_alignment[0]
    best_event_setting = sorted_by_event[0]
    best_alignment_result = evaluate_mhavok_configuration(
        Y,
        t,
        dt,
        switch_times,
        int(best_alignment_setting["delays"]),
        int(best_alignment_setting["rank"]),
        return_series=True,
        event_tolerance=event_tolerance,
    )
    best_event_result = evaluate_mhavok_configuration(
        Y,
        t,
        dt,
        switch_times,
        int(best_event_setting["delays"]),
        int(best_event_setting["rank"]),
        return_series=True,
        event_tolerance=event_tolerance,
    )

    component_r2_df = pd.DataFrame(
        {
            "component": [
                f"v{index + 1}" for index in range(len(best_alignment_result["component_r2"]))
            ],
            "R²": best_alignment_result["component_r2"],
        }
    )

    summary_df, gap_summary_df, selection_df = summarize_best_observable_models(
        X,
        t,
        dt,
        switch_times,
        delay_grid,
        rank_grid,
    )
    combo_best_df = summarize_all_channel_combos(
        X,
        t,
        dt,
        switch_times,
        delay_grid,
        rank_grid,
        event_tolerance,
    )

    export_results(
        Path("plots") / "mhavok_lorenz",
        summary_df,
        gap_summary_df,
        selection_df,
        sweep_metrics_df,
        combo_best_df,
        component_r2_df,
    )
    output_dir = Path("plots") / "mhavok_lorenz"
    save_rank_delay_heatmaps(sweep_metrics_df.to_dict("records"), delay_grid, rank_grid, output_dir)
    save_xz_model_comparison_plot(baseline_result, best_alignment_result, best_event_result, output_dir)
    save_channel_combo_plot(combo_best_df, output_dir)
    save_component_r2_plot(component_r2_df, output_dir)

    print("Baseline x+z configuration:")
    print(
        f"  delays={baseline_delays}, rank={baseline_rank}, "
        f"median |gap|={baseline_result['median_abs_gap']:.4f} s"
    )
    print()
    print("Best x+z setting by switch alignment:")
    print(
        f"  delays={best_alignment_setting['delays']}, rank={best_alignment_setting['rank']}, "
        f"median |gap|={best_alignment_setting['median_abs_gap']:.4f} s, "
        f"fraction within 0.10 s={best_alignment_setting['fraction_within_0p10']:.3f}"
    )
    print()
    print("Best x+z setting by event metrics:")
    print(
        f"  delays={best_event_setting['delays']}, rank={best_event_setting['rank']}, "
        f"recall={best_event_setting['event_recall']:.3f}, "
        f"precision={best_event_setting['event_precision']:.3f}, "
        f"F1={best_event_setting['event_f1']:.3f}, "
        f"chamfer={best_event_setting['chamfer_distance']:.4f} s"
    )
    print()
    print("Best observable combination by event F1:")
    print(combo_best_df.round(4).iloc[0].to_string())
    print()
    print("Exported CSV summaries and plots to plots/mhavok_lorenz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone Lorenz mHAVOK analysis.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller trajectory and smaller sweep for a fast smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(quick=args.quick)


if __name__ == "__main__":
    main()