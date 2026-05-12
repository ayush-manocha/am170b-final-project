---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: "Avenir Next", "Aptos", sans-serif;
    background: #f7f4ed;
    color: #162126;
    padding: 52px;
  }

  h1, h2, h3 {
    color: #143642;
    margin-bottom: 0.25em;
  }

  h1 {
    font-size: 2.1em;
    letter-spacing: 0.02em;
  }

  h2 {
    font-size: 1.45em;
  }

  p, li, table {
    font-size: 0.88em;
    line-height: 1.35;
  }

  strong {
    color: #8a3b12;
  }

  code {
    background: #ece6d8;
    color: #143642;
    padding: 0.1em 0.3em;
    border-radius: 0.2em;
  }

  .lead {
    background: linear-gradient(140deg, #143642 0%, #24565f 55%, #d8b66b 100%);
    color: #f8f5ef;
  }

  .lead h1,
  .lead h2,
  .lead strong,
  .dark h1,
  .dark h2,
  .dark strong {
    color: #fff7e8;
  }

  .lead code,
  .dark code {
    background: rgba(255, 247, 232, 0.14);
    color: #fff7e8;
  }

  .dark {
    background: linear-gradient(160deg, #17313a 0%, #1f4d52 100%);
    color: #edf3ef;
  }

  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }

  .cols-3070 {
    display: grid;
    grid-template-columns: 0.9fr 1.4fr;
    gap: 22px;
    align-items: start;
  }

  .cols-7030 {
    display: grid;
    grid-template-columns: 1.4fr 0.9fr;
    gap: 22px;
    align-items: start;
  }

  .card {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(20, 54, 66, 0.12);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 14px 32px rgba(20, 54, 66, 0.08);
  }

  .dark .card,
  .lead .card {
    background: rgba(255, 247, 232, 0.11);
    border: 1px solid rgba(255, 247, 232, 0.16);
    box-shadow: none;
  }

  .kpi {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 10px;
  }

  .kpi .box {
    background: rgba(20, 54, 66, 0.08);
    border-radius: 14px;
    padding: 10px 12px;
  }

  .dark .kpi .box,
  .lead .kpi .box {
    background: rgba(255, 247, 232, 0.12);
  }

  .box .label {
    display: block;
    font-size: 0.6em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.78;
  }

  .box .value {
    display: block;
    font-size: 1.25em;
    font-weight: 700;
    margin-top: 2px;
  }

  .caption {
    font-size: 0.64em;
    color: #56646b;
    margin-top: 6px;
  }

  .dark .caption,
  .lead .caption {
    color: rgba(248, 245, 239, 0.82);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
  }

  th {
    text-align: left;
    border-bottom: 2px solid rgba(20, 54, 66, 0.28);
    padding: 6px 8px;
    font-size: 0.7em;
    color: #143642;
  }

  td {
    border-bottom: 1px solid rgba(20, 54, 66, 0.12);
    padding: 6px 8px;
    font-size: 0.72em;
  }

  .dark th,
  .lead th {
    color: #fff7e8;
    border-bottom: 2px solid rgba(255, 247, 232, 0.28);
  }

  .dark td,
  .lead td {
    border-bottom: 1px solid rgba(255, 247, 232, 0.14);
  }

  .small {
    font-size: 0.72em;
  }

  .tight li {
    margin: 0.2em 0;
  }

  img.plot {
    width: 100%;
    border-radius: 16px;
    box-shadow: 0 16px 34px rgba(20, 54, 66, 0.12);
    border: 1px solid rgba(20, 54, 66, 0.1);
  }
---

<!-- _class: lead -->

# HAVOK Burst Warning and mHAVOK Model Selection
## Final project results for Hindmarsh-Rose and Lorenz

<div class="cols">
<div class="card">

### Hindmarsh-Rose

- Question: does the HAVOK forcing signal switch on before burst onset?
- Result: **yes for onset warning**, with strong recall across practical lead windows.

</div>
<div class="card">

### Lorenz

- Question: how sensitive is mHAVOK model ranking to rank, delays, and observable choice?
- Result: **very sensitive**. The winner changes with the objective and channel set.

</div>
</div>

<div class="kpi">
<div class="box"><span class="label">HR bursts</span><span class="value">216</span></div>
<div class="box"><span class="label">Best HR recall @ 20 tu</span><span class="value">1.000</span></div>
<div class="box"><span class="label">Best Lorenz combo</span><span class="value">x+y+z</span></div>
</div>

---

# Hindmarsh-Rose Bursting Regime

<div class="cols-3070">
<div class="card tight">

### Setup summary

- Total simulated time: **49049.9** time units
- Bursts detected: **216**
- Mean inter-burst interval: **227.8**
- IBI standard deviation: **81.4**
- Coefficient of variation: **0.36**

### Why this matters

- The target is **burst onset**, not generic large amplitude activity.
- The regime is structured enough for delay embedding, but irregular enough to make onset timing nontrivial.

</div>
<div>

![w:1000](../plots/havok_hr_burst_detection.png)

<div class="caption">Burst detection trace from the HR simulation.</div>

</div>
</div>

---

# HR Forcing Works as an Early-Warning Signal

<div class="cols">
<div>

![w:1000](../plots/havok_hr_recall_vs_window.png)

<div class="caption">Recall remains perfect across 50, 30, 20, and 10 time-unit windows.</div>

</div>
<div class="card tight">

### Core result

- Forcing threshold: **0.001931**
- Active fraction: **0.203**
- Predicted burst onsets: **309**
- Mean lead time: **8.0** time units

### Interpretation

- Every true burst is preceded by forcing activation over practical lead windows.
- Precision is **0.699** at 50, 30, 20, and 10 time units, so the signal is selective enough to be useful even though it is not perfectly sparse.
- The 5-time-unit precision collapse is a timing-resolution issue near onset, not a total failure of the forcing signal.

</div>
</div>

---

# HR Sensitivity to Rank, Delays, and Channel

<div class="cols">
<div>

![w:1000](../plots/havok_hr_rank_delay_sensitivity.png)

<div class="caption">Best x-channel setting on the tested grid: <code>delays = 150</code>, <code>rank = 15</code>.</div>

</div>
<div class="card tight">

### Best tested settings

| Channel | Delays | Rank | Recall | Precision | F1 | Mean linear R^2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x | 150 | 15 | 1.000 | 0.867 | 0.929 | 0.946 |
| y | 150 | 15 | 1.000 | 0.867 | 0.929 | 0.938 |
| z | 150 | 15 | 1.000 | 0.848 | 0.918 | 0.972 |

### Takeaway

- The warning result is not coming from one fragile setting.
- **x and y** are best for event detection on the tested grid.
- **z** gives the strongest linear-state fit, but slightly weaker event precision.

</div>
</div>

---

<!-- _class: dark -->

# HR Caveat: Warning Success Is Not Full-State Success

<div class="cols">
<div class="card tight">

### Strong warning result

- Recall at 20 time units: **1.000**
- Precision at 20 time units: **0.699**
- Lead time: **8.0** time units on average

### Weak generative result

- Forcing-mode R^2: **0.5128**
- Train reconstruction R^2_rec: **-1935.3039**
- Test reconstruction R^2_rec: **-518.8316**

</div>
<div class="card tight">

### Statistical mismatch in free-running behavior

- True IBI mean: **227.8**
- Predicted IBI mean: **159.0**
- KS statistic: **0.3191**, p-value **0.0000**
- True exceedances above 1.5: **6360**
- Reconstructed exceedances above 1.5: **145401**

### Claim we can defend

- HAVOK is useful here as an **event-warning detector**.
- It is **not yet** a reliable free-running surrogate for burst statistics or tail-risk claims.

</div>
</div>

---

# Lorenz: The Best Model Depends on the Objective

<div class="card tight">

### Default <code>x+z</code> setup on the tested rank-delay grid

| Objective | Delays | Rank | Median abs gap (s) | Recall | Precision | F1 | Mean linear R^2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best event F1 | 150 | 11 | 0.0480 | 0.6429 | 0.2535 | 0.3636 | 0.9382 |
| Best alignment | 150 | 13 | 0.0275 | 0.6071 | 0.2208 | 0.3238 | 0.9493 |

</div>

<div class="cols" style="margin-top: 18px;">
<div class="card tight">

### What changed

- The notebook sweep sorts alignment and event metrics separately.
- Those two objectives **do not pick the same model**.

</div>
<div class="card tight">

### Presentation consequence

- The baseline-versus-tuned overlay in the notebook is showing the **best alignment** model.
- It should not be described as the **best event-detection** model.

</div>
</div>

---

# Lorenz Observable Choice Changes the Ranking

<div class="card tight">

| Channel combo | Delays | Rank | Recall | Precision | Accuracy | F1 | Mean linear R^2 | Chamfer (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x+y+z | 150 | 5 | 0.7143 | 0.4878 | 0.8924 | 0.5797 | 0.7450 | 0.1488 |
| x+y | 150 | 5 | 0.6429 | 0.3273 | 0.8794 | 0.4337 | 0.9988 | 0.2095 |
| x | 150 | 5 | 0.6429 | 0.3214 | 0.8792 | 0.4286 | 0.9995 | 0.1937 |
| x+z | 150 | 11 | 0.6429 | 0.2535 | 0.8801 | 0.3636 | 0.9382 | 0.2352 |

</div>

<div class="cols" style="margin-top: 16px;">
<div class="card tight">

### Main result

- The best event-detection model on the tested grid is **x+y+z**, not the earlier x-only or x+z variants.

</div>
<div class="card tight">

### Why this matters

- Adding <code>y</code> changes the ranking materially.
- The strongest event model does **not** maximize mean linear R^2.

</div>
</div>

---

# Lorenz Diagnostics: Most Modes Fit Well, the Last One Does Not

<div class="cols-7030">
<div class="card tight">

### Tuned component fit summary

| Component block | R^2 |
| --- | ---: |
| v1 through v7 | 1.0000 |
| v8 | 0.9991 |
| v9 | 0.9989 |
| v10 | 0.9992 |
| v11 | 0.9986 |
| v12 | 0.3957 |

### Interpretation

- The linear latent structure is fit extremely well for most modes.
- The last forcing-like coordinate is the bottleneck.
- That is why model ranking changes once the evaluation target shifts from fit quality to event detection.

</div>
<div class="card tight small">

### Related Lorenz objective table

| Observable set | Tight alignment | Balanced | Early warning |
| --- | --- | --- | --- |
| z only | winner |  |  |
| x and z |  | winner |  |
| x only |  |  | winner |

This earlier objective-selection table and the full combo benchmark tell the same story: **the winner depends on the objective**.

</div>
</div>

---

<!-- _class: dark -->

# Final Takeaways

<div class="cols">
<div class="card tight">

### Hindmarsh-Rose

- HAVOK forcing is a **strong early-warning signal** for burst onset.
- The result is robust across the tested windows and across multiple input channels.
- The current model should **not** yet be presented as a high-fidelity free-running burst surrogate.

</div>
<div class="card tight">

### Lorenz

- In mHAVOK, **rank, delay, objective, and observable choice all matter**.
- The best alignment model, best event-F1 model, and best full observable set are different.
- The practical lesson is to choose the model **based on the question you want to answer**.

</div>
</div>

<div class="kpi">
<div class="box"><span class="label">HR best tested precision</span><span class="value">0.867</span></div>
<div class="box"><span class="label">Lorenz best combo F1</span><span class="value">0.580</span></div>
<div class="box"><span class="label">Most honest summary</span><span class="value">warning yes, surrogate not yet</span></div>
</div>