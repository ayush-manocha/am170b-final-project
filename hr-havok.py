"""
HAVOK analysis on the Hindmarsh-Rose neuron model.
Evaluates how well the HAVOK forcing term predicts chaotic bursting.

Parameters from:
    Dynamical phases of the Hindmarsh-Rose neuronal model (Innocenti et al. 2007)
    r=0.0021, I_ext=3.281 produces chaotic bursting with positive Lyapunov exponent.

Quality metrics from:
    Colchero et al. (2025) - A multichannel generalization of the HAVOK method
    R² per component (Eq. 8) and R²_rec reconstruction quality (Eq. 19-20).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks, lsim, StateSpace
from scipy.stats import genextreme, ks_2samp, genpareto
from pydmd import HAVOK


def hindmarsh_rose(
    t,
    state,
    a = 1.0,
    b = 3.0,
    c = 1.0,
    d = 5.0,
    r = 0.0021,
    s = 4.0,
    x_rest = -1.6,
    I_ext = 3.281,
):
    """
    Hindmarsh-Rose neuron model:
        dx/dt = y - a*x^3 + b*x^2 - z + I_ext
        dy/dt = c - d*x^2 - y
        dz/dt = r * (s*(x - x_rest) - z)

    Default parameters produce chaotic bursting.
    """
    x, y, z = state
    dx = y - a * x**3 + b * x**2 - z + I_ext
    dy = c - d * x**2 - y
    dz = r * (s * (x - x_rest) - z)
    return [dx, dy, dz]


def generate_hr_data(
    t_eval,
    **kwargs
):
    """Integrate the Hindmarsh-Rose system and return the state matrix."""
    sol = solve_ivp(
        lambda t, s: hindmarsh_rose(t, s, **kwargs),
        [t_eval[0], t_eval[-1]],
        y0 = [0.0, 0.0, 0.0],
        t_eval = t_eval,
        method = "RK45",
        rtol = 1e-10,
        atol = 1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.y


def get_burst_indices(
    x,
    dt,
    height=0.0,
    min_spike_gap = 5.0,
    min_burst_gap = 50.0
):
    """
    Detect burst onset indices.

    1. Find all individual spike peaks above `height`
    2. Group spikes into bursts by inter-spike gap
    3. For each burst, walk backward from the first spike to find
       when x first crossed `height` upward — the true burst start

    Args:
        x (np.ndarray): Membrane potential time series.
        dt (float): Time step.
        height (float): Threshold above which x is considered active.
        min_spike_gap (float): Minimum time between spikes (time units).
        min_burst_gap (float): Inter-spike gap that signals a new burst (time units).

    Returns:
        np.ndarray: Indices of burst onsets (true start, not spike peak).
    """
    spike_idx, _ = find_peaks(x, height=height, distance=int(min_spike_gap / dt))

    if len(spike_idx) == 0:
        return np.array([], dtype=int)

    # Group spikes into bursts
    gap_samples = int(min_burst_gap / dt)
    first_spikes = [spike_idx[0]]
    for i in range(1, len(spike_idx)):
        if spike_idx[i] - spike_idx[i - 1] > gap_samples:
            first_spikes.append(spike_idx[i])

    # Walk backward from each first spike to find true burst start
    burst_onsets = []
    for spike in first_spikes:
        search_start = max(0, spike - int(50 / dt))
        segment = x[search_start:spike]
        above_seg = (segment > height).astype(int)
        crossings_seg = np.where(np.diff(above_seg) == 1)[0]
        if len(crossings_seg) > 0:
            burst_onsets.append(search_start + crossings_seg[-1])
        else:
            burst_onsets.append(spike)  # fallback to spike peak

    return np.array(burst_onsets)


def get_forcing_burst_indices(
    forcing,
    dt,
    threshold,
    min_burst_gap = 30.0
):
    """
    Find the first threshold crossing of each contiguous active region
    in the forcing signal.

    Args:
        forcing (np.ndarray): Forcing time series.
        dt (float): Time step in seconds.
        threshold (float): Activation threshold for |forcing|.
        min_burst_gap (float): Minimum time between active regions (time units).

    Returns:
        np.ndarray: Indices of predicted burst onsets.
    """
    active = np.abs(forcing) > threshold
    # Find transitions from inactive to active
    transitions = np.diff(active.astype(int))

    onsets = np.where(transitions == 1)[0] + 1 # rising edges
    offsets = np.where(transitions == -1)[0] + 1 # falling edges

    if active[0]:
        onsets = np.insert(onsets, 0, 0)
    if active[-1]:
        offsets = np.append(offsets, len(forcing) - 1)

    if len(onsets) == 0:
        return onsets

    gap_samples = int(min_burst_gap / dt)
    merged = [onsets[0]]
    for i in range(1, len(onsets)):
        gap = onsets[i] - offsets[i - 1]
        if gap > gap_samples:
            merged.append(onsets[i])

    return np.array(merged)


def compute_component_r2(
    havok,
):
    """
    Compute R^2 for each embedding component's linear regression fit.
    (Eq. 8 in Colchero et al. 2025)

    Linear components should have R^2 close to 1.
    Nonlinear (forcing) components should have low R^2.

    Args:
        havok: fitted HAVOK instance.

    Returns:
        np.ndarray: R^2 values for each component.
    """
    V = havok.delay_embeddings
    dt = havok.time[1] - havok.time[0]
    V_dot = np.gradient(V, dt, axis = 0)

    r_squared = []
    for i in range(V.shape[1]):
        v_dot_i = V_dot[:, i]
        coeffs = np.linalg.lstsq(V, v_dot_i, rcond = None)[0]
        v_dot_pred = V @ coeffs
        ss_res = np.sum((v_dot_i - v_dot_pred) ** 2)
        ss_tot = np.sum((v_dot_i - v_dot_i.mean()) ** 2)
        r_squared.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    return np.array(r_squared)


def project_to_embedding(
    havok_train,
    x_data
):
    """
    Project data into the embedding space learned from training data.

    Args:
        havok_train: HAVOK model fitted on training data.
        x_data (np.ndarray): Time series to project.

    Returns:
        tuple: (V_linear, V_forcing) - embeddings split into linear
            dynamics and forcing components.
    """
    H = havok_train.hankel(x_data[np.newaxis, :])
    U = havok_train._singular_vecs
    s = havok_train._singular_vals
    V_data = np.linalg.multi_dot([np.diag(1 / s), U.T, H]).T

    num_chaos = havok_train._num_chaos
    V_linear = V_data[:, :-num_chaos]
    V_forcing = V_data[:, -num_chaos:]

    return V_linear, V_forcing


def simulate_havok(
    havok_train,
    V_linear,
    V_forcing,
    t_vec
):
    """
    Simulate the HAVOK linear system using training matrices A and B.

    Args:
        havok_train: HAVOK model fitted on training data.
        V_linear (np.ndarray): True linear dynamics.
        V_forcing (np.ndarray): Forcing terms.
        t_vec (np.ndarray): Time vector corresponding to the data.

    Returns:
        np.ndarray: Simulated linear dynamics.
    """
    A = havok_train.A
    B = havok_train.B
    C = np.eye(len(A))
    D = 0.0 * B
    sys = StateSpace(A, B, C, D)

    t_sim = t_vec[:len(V_forcing)] - t_vec[0]
    V_sim = lsim(sys, U = V_forcing, T = t_sim, X0 = V_linear[0])[1]

    return V_sim


def compute_r2_rec(
    V_true,
    V_sim
):
    """
    Compute reconstruction quality R^2_rec between true and simulated embeddings.
    (Eq. 19-20 in Colchero et. al 2025)

    Args:
        V_true (np.ndarray): True linear dynamics (n, r-1).
        V_sim (np.ndarray): Simulated linear dynamics (n, r-1).

    Returns:
        float: Mean R^2_rec across all embedding dimensions.
    """
    n = min(len(V_true), len(V_sim))
    r2_per_dim = []
    for i in range(V_true.shape[1]):
        v_og = V_true[:n, i]
        v_rec = V_sim[:n, i]
        ss_res = np.sum((v_og - v_rec) ** 2)
        ss_tot = np.sum((v_og - v_rec.mean()) ** 2)
        r2_per_dim.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(r2_per_dim))


def compute_r2_rec_vs_length(
    V_true,
    V_sim,
    t_vec,
    n_points = 20
):
    """
    Compute R^2_rec as a function of simulation length to show
    how reconstruction quality changes over time.

    Args:
        V_true (np.ndarray): True embeddings.
        V_sim (np.ndarray): Simulated embeddings.
        t_vec (np.ndarray): Time vector.
        n_points (int): Number of length values to evaluate.

    Returns:
        tuple: (times, r2_values)
    """
    n_max  = min(len(V_true), len(V_sim))
    fracs  = np.linspace(0.05, 1.0, n_points)
    times  = []
    r2s    = []
    for frac in fracs:
        n = max(int(frac * n_max), 10)
        r2 = compute_r2_rec(V_true[:n], V_sim[:n])
        times.append(t_vec[n - 1] - t_vec[0])
        r2s.append(r2)
    return np.array(times), np.array(r2s)


def compute_rolling_rmse(
    V_true,
    V_sim,
    t_vec,
    window_tu = 50.0,
    dt = 0.1
):
    """
    Compute rolling RMSE between true and simulated embeddings.

    Args:
        V_true (np.ndarray): True embeddings.
        V_sim (np.ndarray): Simulated embeddings.
        t_vec (np.ndarray): Time vector.
        window_tu (float): Rolling window size in time units.
        dt (float): Time step.

    Returns:
        tuple: (times, rmse_values)
    """
    window = int(window_tu / dt)
    n = min(len(V_true), len(V_sim))
    times, rmse_values = [], []
    for i in range(0, n - window, window // 2):
        chunk_true = V_true[i:i + window]
        chunk_sim = V_sim[i:i + window]
        rmse = np.sqrt(np.mean((chunk_true - chunk_sim) ** 2))
        times.append(t_vec[i] - t_vec[0])
        rmse_values.append(rmse)
    return np.array(times), np.array(rmse_values)


def analyze_ibi_distribution(
    true_onsets,
    pred_onsets,
    t,
    dt
):
    """
    Compute and compare inter-burst interval distributions between
    true and predicted burst onsets.

    Fits a Generalized Extreme Value (GEV) distribution to both,
    computes a KS test to measure similarity, and plots the comparison.

    Args:
        true_onsets (np.ndarray): Indices of true burst onsets.
        pred_onsets (np.ndarray): Indices of predicted burst onsets.
        t (np.ndarray): Time vector.
        dt (float): Time step.
    """
    # Compute IBIs in time units
    ibi_true = np.diff(t[true_onsets])
    ibi_pred = np.diff(t[pred_onsets])

    # Fit GEV distribution to both
    shape_true, loc_true, scale_true = genextreme.fit(ibi_true)
    shape_pred, loc_pred, scale_pred = genextreme.fit(ibi_pred)

    # KS test between true and predicted IBI distributions
    ks_stat, ks_pval = ks_2samp(ibi_true, ibi_pred)

    print(f"\n── IBI distribution analysis ─────────────────────────────────────")
    print(f"  True IBI:  n={len(ibi_true)}  mean={ibi_true.mean():.1f}"
          f"  std={ibi_true.std():.1f}  min={ibi_true.min():.1f}"
          f"  max={ibi_true.max():.1f}")
    print(f"  Pred IBI:  n={len(ibi_pred)}  mean={ibi_pred.mean():.1f}"
          f"  std={ibi_pred.std():.1f}  min={ibi_pred.min():.1f}"
          f"  max={ibi_pred.max():.1f}")
    print(f"  GEV fit (true): shape={shape_true:.3f}  loc={loc_true:.3f}"
          f"  scale={scale_true:.3f}")
    print(f"  GEV fit (pred): shape={shape_pred:.3f}  loc={loc_pred:.3f}"
          f"  scale={scale_pred:.3f}")
    print(f"  KS statistic: {ks_stat:.4f}  p-value: {ks_pval:.4f}")
    print(f"  {'Distributions are similar (p>0.05)' if ks_pval > 0.05 else 'Distributions differ significantly (p<0.05)'}")

    # Plot
    x_range = np.linspace(
        min(ibi_true.min(), ibi_pred.min()),
        max(ibi_true.max(), ibi_pred.max()),
        500
    )
    pdf_true = genextreme.pdf(x_range, shape_true, loc_true, scale_true)
    pdf_pred = genextreme.pdf(x_range, shape_pred, loc_pred, scale_pred)

    fig, axes = plt.subplots(1, 2, figsize = (12, 4))

    # (a) Histogram + fitted GEV
    ax = axes[0]
    ax.hist(ibi_true, bins=30, density=True, alpha=0.5,
            color="tab:blue", edgecolor="k", label="True IBI")
    ax.hist(ibi_pred, bins=30, density=True, alpha=0.5,
            color="tab:orange", edgecolor="k", label="Predicted IBI")
    ax.plot(x_range, pdf_true, c="tab:blue",   lw=2, label="GEV fit (true)")
    ax.plot(x_range, pdf_pred, c="tab:orange", lw=2, ls="--",
            label="GEV fit (pred)")
    ax.set_xlabel("Inter-burst interval (time units)")
    ax.set_ylabel("Density")
    ax.set_title(f"IBI distribution  |  KS p={ks_pval:.3f}")
    ax.legend(fontsize=8)

    # (b) Return period plot
    # Sort IBIs and compute empirical return period
    ax = axes[1]
    for ibi, color, label in [
        (ibi_true, "tab:blue",   "True"),
        (ibi_pred, "tab:orange", "Predicted"),
    ]:
        sorted_ibi = np.sort(ibi)
        n          = len(sorted_ibi)
        # Empirical return period = mean IBI / exceedance probability
        exceedance = 1 - np.arange(1, n + 1) / (n + 1)
        return_period = 1 / exceedance
        ax.plot(return_period, sorted_ibi, "o", c=color,
                ms=3, alpha=0.6, label=f"{label} (empirical)")

        # GEV fitted return period
        p_range = np.linspace(0.01, 0.99, 200)
        rp_range = 1 / (1 - p_range)
        quantiles = genextreme.ppf(p_range, *genextreme.fit(ibi))
        ax.plot(rp_range, quantiles, c=color, lw=2)

    ax.set_xscale("log")
    ax.set_xlabel("Return period")
    ax.set_ylabel("Inter-burst interval (time units)")
    ax.set_title("Return period of inter-burst intervals")
    ax.legend(fontsize=8)

    plt.suptitle("IBI distribution — true vs HAVOK predicted", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/havok_hr_ibi_distribution.png", dpi=150)
    plt.show()


def analyze_extremes(x_true, x_recon, t, dt, threshold=1.5):
    """
    Extreme value analysis of membrane potential x.
    Compares exceedance statistics between true and reconstructed x.
    
    Args:
        x_true (np.ndarray): True membrane potential.
        x_recon (np.ndarray): HAVOK reconstructed membrane potential.
        t (np.ndarray): Time vector.
        dt (float): Time step.
        threshold (float): Exceedance threshold.
    """
    n = min(len(x_true), len(x_recon))
    x_true  = x_true[:n]
    x_recon = x_recon[:n]
    t_plot  = t[:n]

    # Exceedances above threshold
    exc_true  = x_true[x_true   > threshold] - threshold
    exc_recon = x_recon[x_recon > threshold] - threshold

    # Rate of exceedance (events per time unit)
    rate_true  = len(exc_true)  / (n * dt)
    rate_recon = len(exc_recon) / (n * dt)

    # Return period in time units = 1 / (rate * exceedance probability)
    # Fit GPD to exceedances (standard peaks-over-threshold approach)
    shape_true,  loc_true,  scale_true  = genpareto.fit(exc_true,  floc=0)
    shape_recon, loc_recon, scale_recon = genpareto.fit(exc_recon, floc=0)

    print(f"\n── Extreme value analysis (threshold = {threshold}) ──────────────")
    print(f"  True exceedances:  n={len(exc_true)}   rate={rate_true:.4f}/tu")
    print(f"  Recon exceedances: n={len(exc_recon)}  rate={rate_recon:.4f}/tu")
    print(f"  GPD fit (true):  shape={shape_true:.3f}  scale={scale_true:.3f}")
    print(f"  GPD fit (recon): shape={shape_recon:.3f}  scale={scale_recon:.3f}")

    # Return period curves
    # P(X > x) = (1 - CDF(x - threshold)) * rate
    # Return period T = 1 / P(X > x)
    exc_range = np.linspace(0, max(exc_true.max(), exc_recon.max()), 300)
    
    def return_period(exc_values, shape, scale, rate):
        surv = 1 - genpareto.cdf(exc_values, shape, loc=0, scale=scale)
        return 1 / (surv * rate)

    rp_true  = return_period(exc_range, shape_true,  scale_true,  rate_true)
    rp_recon = return_period(exc_range, shape_recon, scale_recon, rate_recon)

    # Empirical return periods
    def empirical_rp(exc, rate):
        sorted_exc = np.sort(exc)
        n          = len(sorted_exc)
        exceedance_prob = (1 - np.arange(1, n+1) / (n+1)) * rate
        return 1 / exceedance_prob, sorted_exc

    emp_rp_true,  emp_exc_true  = empirical_rp(exc_true,  rate_true)
    emp_rp_recon, emp_exc_recon = empirical_rp(exc_recon, rate_recon)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) PDF of exceedances
    ax = axes[0]
    ax.hist(exc_true  + threshold, bins=30, density=True, alpha=0.5,
            color="tab:blue",   edgecolor="k", label="True x")
    ax.hist(exc_recon + threshold, bins=30, density=True, alpha=0.5,
            color="tab:red",    edgecolor="k", label="Reconstructed x")
    ax.axvline(threshold, c="k", ls="--", lw=1.5, label=f"Threshold = {threshold}")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of extreme x values")
    ax.legend(fontsize=8)

    # (b) Return period
    ax = axes[1]
    ax.plot(emp_rp_true,  emp_exc_true  + threshold, "o",
            c="tab:blue", ms=3, alpha=0.5, label="True (empirical)")
    ax.plot(emp_rp_recon, emp_exc_recon + threshold, "o",
            c="tab:red",  ms=3, alpha=0.5, label="Reconstructed (empirical)")
    ax.plot(rp_true,  exc_range + threshold,
            c="tab:blue", lw=2, label="True (GPD fit)")
    ax.plot(rp_recon, exc_range + threshold,
            c="tab:red",  lw=2, ls="--", label="Reconstructed (GPD fit)")
    ax.set_xscale("log")
    ax.set_xlabel("Return period (time units)")
    ax.set_ylabel("x")
    ax.set_title("Return period of extreme membrane potential")
    ax.legend(fontsize=8)

    plt.suptitle("Extreme value analysis — true vs HAVOK reconstruction",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/havok_hr_extremes.png", dpi=150)
    plt.show()


def compute_recall_vs_window(
    true_onsets,
    forcing,
    dt,
    threshold,
    windows = [50.0, 30.0, 20.0, 10.0, 5.0],
    n_random = 10000
):
    """
    For each prediction window size, compute:
    - HAVOK recall: fraction of bursts with forcing active in window before onset
    - Baseline recall: same for random windows (empirical chance level)

    Args:
        true_onsets (np.ndarray): Indices of true burst onsets.
        forcing (np.ndarray): Forcing time series.
        dt (float): Time step.
        threshold (float): Activation threshold.
        windows (list): Prediction window sizes in time units.
        n_random (int): Number of random windows for baseline.

    Returns:
        list[dict]: List of dicts with window, recall, baseline.
    """
    results = []
    for window in windows:
        window_samples = int(window / dt)

        # HAVOK recall
        TP = FN = 0
        for i in range(len(true_onsets) - 1):
            burst_start = true_onsets[i + 1]
            window_start = burst_start - window_samples
            f_start = max(0, min(window_start, len(forcing) - 1))
            f_end = min(burst_start, len(forcing))
            if np.any(np.abs(forcing[f_start:f_end]) > threshold):
                TP += 1
            else:
                FN += 1

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        # Baseline: random windows
        rand_starts = np.random.randint(0, len(forcing) - window_samples, n_random)
        baseline = np.mean([
            np.any(np.abs(forcing[i:i + window_samples]) > threshold)
            for i in rand_starts
        ])

        results.append(dict(window = window, TP = TP, FN = FN, recall = recall, baseline = baseline))

    return results


def compute_precision_vs_window(
    true_onsets,
    pred_onsets,
    t,
    dt,
    windows = [50.0, 30.0, 20.0, 10.0, 5.0],
):
    """
    For each prediction window size, compute precision:
    fraction of predicted onsets followed by a true burst within the window.
 
    Args:
        true_onsets (np.ndarray): Indices of true burst onsets.
        pred_onsets (np.ndarray): Indices of predicted burst onsets.
        t (np.ndarray): Time vector.
        dt (float): Time step.
        windows (list): Prediction window sizes in time units.
 
    Returns:
        list[dict]: List of dicts with window, precision, TP, FP.
    """
    results = []
    for window in windows:
        window_samples = int(window / dt)
        TP = FP = 0
        for p in pred_onsets:
            future_true = true_onsets[true_onsets >= p]
            if len(future_true) > 0 and (future_true[0] - p) <= window_samples:
                TP += 1
            else:
                FP += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        results.append(dict(window=window, TP=TP, FP=FP, precision=precision))
    return results


def compute_lead_times(
    true_onsets,
    pred_onsets,
    t,
    max_lead = None
):
    """
    For each burst, find the last predicted onset before it and compute the lead time.
    Optionally cap at max_lead to exclude contamination from previous burst activity.

    Args:
        true_onsets (np.ndarray): Indices of true burst onsets.
        pred_onsets (np.ndarray): Indices of predicted burst onsets.
        t (np.ndarray): Time vector.
        max_lead (float): Maximum valid lead time.

    Returns:
        np.ndarray: Lead times in time units.
    """
    lead_times = []
    for true_idx in true_onsets:
        preds_before = pred_onsets[pred_onsets < true_idx]
        if len(preds_before) > 0:
            lead = t[true_idx] - t[preds_before[-1]]
            if max_lead is None or lead <= max_lead:
                lead_times.append(lead)
    return np.array(lead_times)


def main():
    # Simulation
    dt = 0.1
    m = 500_000
    t = np.arange(m) * dt

    print("Simulating Hindmarsh-Rose system...")
    X = generate_hr_data(t)

    # Discard transient
    transient = int(950 / dt)
    x = X[0, transient:]
    t = t[transient:] - t[transient]

    # Ground-truth bursts
    true_onsets = get_burst_indices(x, dt)
    ibi = np.diff(t[true_onsets])

    print(f"\n── System characterization ───────────────────────────────────────")
    print(f"  Total time:          {t[-1]:.1f} time units")
    print(f"  Bursts detected:     {len(true_onsets)}")
    print(f"  IBI mean:            {ibi.mean():.1f}")
    print(f"  IBI std:             {ibi.std():.1f}  (cv = {ibi.std()/ibi.mean():.2f})")
    print(f"  IBI min:             {ibi.min():.1f}")
    print(f"  IBI max:             {ibi.max():.1f}")

    # HAVOK fit on full data (for burst prediction)
    print("Fitting HAVOK model...")
    havok = HAVOK(svd_rank = 15, delays = 100, num_chaos = 1)
    havok.fit(x, t)

    forcing = havok.forcing[:, 0]
    forcing_time = t[:len(forcing)]
    threshold = havok.compute_threshold(forcing=0, p=0.2)
    pred_onsets = get_forcing_burst_indices(forcing, dt, threshold)

    # Compare true and predicted inter-burst interval distributions
    analyze_ibi_distribution(true_onsets, pred_onsets, t, dt)

    x_recon = havok.reconstructed_data
    analyze_extremes(x, x_recon, t, dt, threshold=1.5)

    # mHAVOK quality metrics
    r2_components = compute_component_r2(havok)

    print(f"\n── HAVOK model quality ──────────────────────────────────────────")
    print(f"  R^2 per component:")
    for i, r2 in enumerate(r2_components):
        tag = " ← forcing (nonlinear)" if i == len(r2_components) - 1 else ""
        print(f"    v{i+1:02d}: {r2:.4f}{tag}")

    # 70/30 train/test split for reconstruction quality
    print("\nFitting HAVOK model (70/30 train/test split)...")
    split = int(0.7 * len(x))
    x_train = x[:split]
    t_train = t[:split]
    x_test = x[split:]
    t_test = t[split:]
 
    havok_train = HAVOK(svd_rank=15, delays=100, num_chaos=1)
    havok_train.fit(x_train, t_train)

    # Project and simulate on training data
    V_linear_train, V_forcing_train = project_to_embedding(havok_train, x_train)
    V_sim_train = simulate_havok(
        havok_train, V_linear_train, V_forcing_train, t_train
    )
    r2_rec_train = compute_r2_rec(V_linear_train, V_sim_train)
    sim_times_train, r2_vs_length_train = compute_r2_rec_vs_length(
        V_linear_train, V_sim_train, t_train
    )
    rmse_times_train, rmse_values_train = compute_rolling_rmse(
        V_linear_train, V_sim_train, t_train, window_tu=50.0, dt=dt
    )
 
    # Project and simulate on test data
    V_linear_test, V_forcing_test = project_to_embedding(havok_train, x_test)
    V_sim_test = simulate_havok(
        havok_train, V_linear_test, V_forcing_test, t_test
    )
    r2_rec_test = compute_r2_rec(V_linear_test, V_sim_test)
    sim_times_test, r2_vs_length_test = compute_r2_rec_vs_length(
        V_linear_test, V_sim_test, t_test
    )
    rmse_times_test, rmse_values_test = compute_rolling_rmse(
        V_linear_test, V_sim_test, t_test, window_tu=50.0, dt=dt
    )
 
    print(f"  R²_rec on train set: {r2_rec_train:.4f}")
    print(f"  R²_rec on test set:  {r2_rec_test:.4f}")

    print(f"V_linear_test range: {V_linear_test.min():.4f} to {V_linear_test.max():.4f}")
    print(f"V_sim_test range:    {V_sim_test.min():.4f} to {V_sim_test.max():.4f}")
    print(f"V_linear_test shape: {V_linear_test.shape}")
    print(f"V_sim_test shape:    {V_sim_test.shape}")

    # Forcing statistics
    q_start = true_onsets[0] + int(50 / dt)
    q_end   = true_onsets[1] - int(50 / dt)
    b_start = true_onsets[0] - int(20 / dt)
    b_end   = true_onsets[0] + int(20 / dt)
    active_fraction = np.mean(np.abs(forcing) > threshold)

    print(f"\n── HAVOK forcing statistics ──────────────────────────────────────")
    print(f"  Threshold:                 {threshold:.6f}")
    print(f"  Forcing std (overall):     {np.std(forcing):.6f}")
    print(f"  Forcing std (quiescent):   {np.std(forcing[q_start:q_end]):.6f}")
    print(f"  Forcing std (burst):       {np.std(forcing[b_start:b_end]):.6f}")
    print(f"  Fraction of time active:   {active_fraction:.3f}")
    print(f"  Predicted burst onsets:    {len(pred_onsets)}")

    # Lead time
    # Cap at half the minimum IBI to exclude contamination from previous burst
    max_lead    = 0.5 * ibi.min()
    lead_times  = compute_lead_times(true_onsets, pred_onsets, t, max_lead=max_lead)
 
    print(f"\n── Lead time (capped at {max_lead:.1f} tu) ────────────────────────")
    print(f"  Bursts with valid lead time: {len(lead_times)}/{len(true_onsets)}")
    print(f"  Mean lead time:              {np.mean(lead_times):.1f} time units")
    print(f"  Std lead time:               {np.std(lead_times):.1f} time units")
    print(f"  Lead time as fraction of IBI: {np.mean(lead_times)/ibi.mean():.2f}")

    # Recall and precision vs window
    windows           = [50.0, 30.0, 20.0, 10.0, 5.0]
    recall_results    = compute_recall_vs_window(true_onsets, forcing, dt, threshold, windows)
    precision_results = compute_precision_vs_window(true_onsets, pred_onsets, t, dt, windows)
 
    print(f"\n── Recall vs prediction window ───────────────────────────────────")
    print(f"  {'Window':>8}  {'Recall':>8}  {'Baseline':>8}  {'Improvement':>12}")
    for r in recall_results:
        print(f"  {r['window']:>8.1f}  {r['recall']:>8.3f}  "
              f"{r['baseline']:>8.3f}  {r['recall']-r['baseline']:>+12.3f}")
 
    print(f"\n── Precision vs prediction window ────────────────────────────────")
    print(f"  {'Window':>8}  {'Precision':>10}  {'TP':>6}  {'FP':>6}")
    for r in precision_results:
        print(f"  {r['window']:>8.1f}  {r['precision']:>10.3f}  "
              f"{r['TP']:>6}  {r['FP']:>6}")

    # Plots
    n_plot  = 15_000
    t_plot  = t[:n_plot]
    x_plot  = x[:n_plot]
    f_plot  = forcing[:n_plot]

    in_range      = true_onsets[true_onsets < n_plot]
    pred_in_range = pred_onsets[pred_onsets < n_plot]

    # (1) Main detection plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("HAVOK on Hindmarsh-Rose — Burst Detection", fontsize=14)
 
    ax = axes[0]
    ax.plot(t_plot, x_plot, c="k", lw=0.6)
    ax.vlines(t[in_range], x_plot.min(), x_plot.max(),
              color="tab:blue", lw=1.2, alpha=0.7, label="True burst onset")
    ax.set_ylabel("x")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(a) Membrane potential")
 
    ax = axes[1]
    ax.plot(forcing_time[:n_plot], f_plot, c="gray", lw=0.7)
    ax.fill_between(forcing_time[:n_plot], f_plot,
                    where=np.abs(f_plot) > threshold,
                    color="tab:red", alpha=0.5, label="Active forcing")
    ax.axhline( threshold, c="tab:red", ls="--", lw=1,
                label=f"±threshold={threshold:.4f}")
    ax.axhline(-threshold, c="tab:red", ls="--", lw=1)
    ax.vlines(t[pred_in_range], f_plot.min(), f_plot.max(),
              color="tab:orange", lw=1.2, alpha=0.9, label="Predicted burst onset")
    ax.set_ylabel("Forcing")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(b) HAVOK forcing term")
 
    ax = axes[2]
    ax.plot(t_plot, x_plot, c="k", lw=0.5, alpha=0.5)
    ax.vlines(t[in_range], x_plot.min(), x_plot.max(),
              color="tab:blue", lw=1.5, alpha=0.7, label="True")
    ax.vlines(t[pred_in_range], x_plot.min(), x_plot.max(),
              color="tab:orange", lw=1.5, alpha=0.7, ls="--", label="Predicted")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(c) Burst alignment")
 
    plt.tight_layout()
    plt.savefig("plots/havok_hr_burst_detection.png", dpi=150)
    plt.show()
 
    # (2) Recall and precision vs window
    recalls    = [r['recall']    for r in recall_results]
    baselines  = [r['baseline']  for r in recall_results]
    precisions = [r['precision'] for r in precision_results]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    ax = axes[0]
    ax.plot(windows, recalls,   "o-",  c="tab:blue", lw=2, label="HAVOK recall")
    ax.plot(windows, baselines, "s--", c="gray",     lw=2, label="Baseline (random)")
    ax.set_xlabel("Prediction window (time units)")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs prediction window")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
 
    ax = axes[1]
    ax.plot(windows, precisions, "o-", c="tab:orange", lw=2, label="HAVOK precision")
    ax.set_xlabel("Prediction window (time units)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs prediction window")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
 
    plt.suptitle("HAVOK burst prediction accuracy vs window size", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/havok_hr_accuracy_vs_window.png", dpi=150)
    plt.show()
 
    # (3) Lead time histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lead_times, bins=20, color="tab:blue", edgecolor="k", alpha=0.7)
    ax.axvline(np.mean(lead_times), c="tab:red", lw=2,
               label=f"Mean = {np.mean(lead_times):.1f} tu")
    ax.axvline(0, c="k", lw=1.5, ls="--", label="Burst start")
    ax.set_xlabel("Lead time (time units)")
    ax.set_ylabel("Count")
    ax.set_title("HAVOK forcing lead time before burst onset")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/havok_hr_lead_times.png", dpi=150)
    plt.show()

    # (4) R^2 per component (mHAVOK quality metric)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["tab:red" if i == len(r2_components) - 1 else "tab:green"
              for i in range(len(r2_components))]
    ax.bar(range(1, len(r2_components) + 1), r2_components,
           color=colors, edgecolor="k", alpha=0.8)
    ax.axhline(0.95, c="k", ls="--", lw=1.5, label="τ = 0.95 (linear threshold)")
    ax.set_xlabel("Component index")
    ax.set_ylabel("R²")
    ax.set_title(f"R² per embedding component")
    ax.set_xticks(range(1, len(r2_components) + 1))
    ax.set_ylim(0, 1.05)
 
    # Add legend patches
    ax.legend(handles=[
        Patch(color="tab:green", label="Linear component"),
        Patch(color="tab:red",   label="Nonlinear / forcing component"),
        plt.Line2D([0], [0], c="k", ls="--", lw=1.5, label="τ = 0.95"),
    ])
    plt.tight_layout()
    plt.savefig("plots/havok_hr_r2_components.png", dpi=150)
    plt.show()

    # (5) Embedding reconstruction — train vs test side by side
    n_show    = int(500 / dt)
    n_show_tr = min(n_show, len(V_sim_train), len(V_linear_train))
    n_show_te = min(n_show, len(V_sim_test),  len(V_linear_test))
 
    fig, axes = plt.subplots(3, 2, figsize=(14, 7), sharex="col")
    fig.suptitle("HAVOK embedding reconstruction — train vs test (first 500 tu)",
                 fontsize=13)
 
    for i in range(3):
        # Train
        t_tr = t_train[:n_show_tr] - t_train[0]
        axes[i, 0].plot(t_tr, V_linear_train[:n_show_tr, i],
                        c="k", lw=0.8, label="True")
        axes[i, 0].plot(t_tr, V_sim_train[:n_show_tr, i],
                        c="gray", lw=0.8, ls="--", label="Reconstructed")
        axes[i, 0].set_ylabel(f"v{i+1}")
        axes[i, 0].set_yticks([])
 
        # Test
        t_te = t_test[:n_show_te] - t_test[0]
        axes[i, 1].plot(t_te, V_linear_test[:n_show_te, i],
                        c="k", lw=0.8, label="True")
        axes[i, 1].plot(t_te, V_sim_test[:n_show_te, i],
                        c="tab:red", lw=0.8, ls="--", label="Reconstructed")
        axes[i, 1].set_yticks([])
 
    axes[0, 0].set_title("Train set")
    axes[0, 1].set_title("Test set")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("Time (time units)")
    axes[-1, 1].set_xlabel("Time (time units)")
 
    plt.tight_layout()
    plt.savefig("plots/havok_hr_reconstruction.png", dpi=150)
    plt.show()
 
    # (6) R²_rec vs simulation length — train and test
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    ax = axes[0]
    ax.plot(sim_times_train, r2_vs_length_train, "o-", c="gray",
            lw=2, alpha=0.8, label=f"Train  (R²_rec = {r2_rec_train:.1f})")
    ax.plot(sim_times_test,  r2_vs_length_test,  "o-", c="tab:blue",
            lw=2, label=f"Test   (R²_rec = {r2_rec_test:.1f})")
    ax.axhline(0, c="k", ls="--", lw=1, label="Baseline (predict mean)")
    ax.set_xlabel("Simulation length (time units)")
    ax.set_ylabel("R²_rec")
    ax.set_title("Reconstruction quality vs simulation length")
    ax.legend(fontsize=8)
 
    ax = axes[1]
    ax.plot(rmse_times_train, rmse_values_train,
            c="gray", lw=1.5, alpha=0.8, label="Train")
    ax.plot(rmse_times_test,  rmse_values_test,
            c="tab:blue", lw=1.5, label="Test")
    ax.set_xlabel("Time (time units)")
    ax.set_ylabel("RMSE (embedding space)")
    ax.set_title("Rolling RMSE of embedding reconstruction")
    ax.legend(fontsize=8)
 
    plt.suptitle("HAVOK reconstruction quality — train vs test", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/havok_hr_reconstruction_quality.png", dpi=150)
    plt.show()
 
    # (7) 3D attractor — true vs reconstructed
    fig = plt.figure(figsize=(12, 5))
 
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(V_linear_test[:, 0], V_linear_test[:, 1], V_linear_test[:, 2],
             c="k", lw=0.3, alpha=0.5)
    ax1.set_title("True embedded attractor (test set)")
    ax1.set_xlabel("v1"); ax1.set_ylabel("v2"); ax1.set_zlabel("v3")
 
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(V_sim_test[:, 0], V_sim_test[:, 1], V_sim_test[:, 2],
             c="tab:red", lw=0.3, alpha=0.5)
    ax2.set_title("Reconstructed attractor (HAVOK simulation)")
    ax2.set_xlabel("v1"); ax2.set_ylabel("v2"); ax2.set_zlabel("v3")
 
    plt.suptitle("Embedded attractor — true vs HAVOK reconstruction", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/havok_hr_attractor.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
