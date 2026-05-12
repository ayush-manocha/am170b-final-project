---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: "Avenir Next", "Aptos", sans-serif;
    background: #f5f1e8;
    color: #162126;
    padding: 52px;
  }

  h1, h2, h3 {
    color: #173c44;
    margin-bottom: 0.28em;
  }

  h1 { font-size: 2.05em; }
  h2 { font-size: 1.42em; }

  p, li, table {
    font-size: 0.86em;
    line-height: 1.34;
  }

  strong {
    color: #8e3b12;
  }

  code {
    background: #ebe3d3;
    color: #173c44;
    padding: 0.1em 0.3em;
    border-radius: 0.2em;
  }

  .lead {
    background: linear-gradient(145deg, #173c44 0%, #215c61 55%, #d6b466 100%);
    color: #f8f5ef;
  }

  .lead h1,
  .lead h2,
  .lead strong,
  .dark h1,
  .dark h2,
  .dark strong {
    color: #fff7e7;
  }

  .lead code,
  .dark code {
    background: rgba(255, 247, 231, 0.12);
    color: #fff7e7;
  }

  .dark {
    background: linear-gradient(160deg, #17313a 0%, #1f4b51 100%);
    color: #eef3ef;
  }

  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }

  .cols-3070 {
    display: grid;
    grid-template-columns: 0.95fr 1.45fr;
    gap: 24px;
    align-items: start;
  }

  .card {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid rgba(23, 60, 68, 0.12);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 14px 32px rgba(23, 60, 68, 0.08);
  }

  .dark .card,
  .lead .card {
    background: rgba(255, 247, 231, 0.10);
    border: 1px solid rgba(255, 247, 231, 0.15);
    box-shadow: none;
  }

  .kpi {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 12px;
  }

  .kpi .box {
    background: rgba(23, 60, 68, 0.08);
    border-radius: 14px;
    padding: 10px 12px;
  }

  .dark .kpi .box,
  .lead .kpi .box {
    background: rgba(255, 247, 231, 0.12);
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
    font-size: 1.22em;
    font-weight: 700;
    margin-top: 3px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
  }

  th {
    text-align: left;
    border-bottom: 2px solid rgba(23, 60, 68, 0.28);
    padding: 6px 8px;
    font-size: 0.7em;
    color: #173c44;
  }

  td {
    border-bottom: 1px solid rgba(23, 60, 68, 0.12);
    padding: 6px 8px;
    font-size: 0.72em;
  }

  .dark th,
  .lead th {
    color: #fff7e7;
    border-bottom: 2px solid rgba(255, 247, 231, 0.28);
  }

  .dark td,
  .lead td {
    border-bottom: 1px solid rgba(255, 247, 231, 0.15);
  }

  .small { font-size: 0.72em; }

  .caption {
    font-size: 0.64em;
    color: #5c696d;
    margin-top: 6px;
  }

  .dark .caption,
  .lead .caption {
    color: rgba(248, 245, 239, 0.82);
  }
---

<!-- _class: lead -->

# mHAVOK on Lorenz
## Parallel event metrics, Chamfer distance, and channel-combo comparison

<div class="cols">
<div class="card">

### Task completed

- Standalone implementation extracted to <code>mhavok_lorenz.py</code>
- Full metrics exported for **recall, precision, accuracy, F1, mean linear R², and Chamfer distance**
- All single-, double-, and triple-channel combinations of <code>x</code>, <code>y</code>, and <code>z</code> benchmarked

</div>
<div class="card">

### Main conclusion

- For the default <code>x+z</code> setup, the best **alignment** model and best **event-F1** model are different.
- Across all channel combinations, the best event-detection model on the tested grid is **<code>x+y+z</code>**.

</div>
</div>

<div class="kpi">
<div class="box"><span class="label">Best x+z event F1</span><span class="value">0.364</span></div>
<div class="box"><span class="label">Best combo</span><span class="value">x+y+z</span></div>
<div class="box"><span class="label">Best combo F1</span><span class="value">0.580</span></div>
</div>

---

# x+z Baseline vs Tuned mHAVOK Models

<div class="cols-3070">
<div class="card">

### Baseline and winners

| Model | Delays | Rank | Recall | Precision | Accuracy | F1 | Mean linear R² | Chamfer (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline x+z | 100 | 9 | 0.1786 | 0.0794 | 0.8653 | 0.1099 | 0.9173 | 0.2236 |
| Best alignment | 150 | 13 | 0.6071 | 0.2208 | 0.8872 | 0.3238 | 0.9493 | 0.2740 |
| Best event model | 150 | 11 | 0.6429 | 0.2535 | 0.8801 | 0.3636 | 0.9382 | 0.2352 |

### Takeaway

- The tuned x+z model clearly improves event metrics over the baseline.
- The **best alignment** model is not the same as the **best event-detection** model.

</div>
<div>

![w:1000](../plots/mhavok_lorenz/mhavok_lorenz_xz_model_comparison.png)

<div class="caption">Direct comparison of baseline, best-alignment, and best-event x+z models on the requested metrics.</div>

</div>
</div>

---

# Rank-Delay Sweep for Event Metrics

<div class="cols-3070">
<div class="card small">

### Best x+z settings on the tested grid

| Objective | Delays | Rank | Recall | Precision | Accuracy | F1 | Mean linear R² | Chamfer (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best event F1 | 150 | 11 | 0.6429 | 0.2535 | 0.8801 | 0.3636 | 0.9382 | 0.2352 |
| Best alignment | 150 | 13 | 0.6071 | 0.2208 | 0.8872 | 0.3238 | 0.9493 | 0.2740 |

### Reading the heatmap

- Event metrics improve strongly at **higher delays**.
- High mean linear R² does not uniquely determine the best event model.
- Chamfer distance and F1 do not select the same point as tight alignment.

</div>
<div>

![w:1000](../plots/mhavok_lorenz/mhavok_lorenz_rank_delay_metrics.png)

<div class="caption">Recall, precision, accuracy, F1, mean linear R², and Chamfer distance over the tested rank-delay grid.</div>

</div>
</div>

---

# Channel Combinations: x, y, z Matter

<div class="cols-3070">
<div class="card small">

### Best model from each channel combination

| Combo | Delays | Rank | Recall | Precision | Accuracy | F1 | Mean linear R² | Chamfer (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x+y+z | 150 | 5 | 0.7143 | 0.4878 | 0.8924 | 0.5797 | 0.7450 | 0.1488 |
| x+y | 150 | 5 | 0.6429 | 0.3273 | 0.8794 | 0.4337 | 0.9988 | 0.2095 |
| x | 150 | 5 | 0.6429 | 0.3214 | 0.8792 | 0.4286 | 0.9995 | 0.1937 |
| x+z | 150 | 11 | 0.6429 | 0.2535 | 0.8801 | 0.3636 | 0.9382 | 0.2352 |

### Main result

- The strongest event-detection model on the tested grid is **x+y+z**.
- Adding <code>y</code> materially changes the ranking.

</div>
<div>

![w:1000](../plots/mhavok_lorenz/mhavok_lorenz_channel_combo_metrics.png)

<div class="caption">Per-combo best recall, precision, accuracy, F1, mean linear R², and Chamfer distance.</div>

</div>
</div>

---

# R² Diagnostics and Objective Tradeoffs

<div class="cols">
<div>

![w:1000](../plots/mhavok_lorenz/mhavok_lorenz_component_r2.png)

<div class="caption">Tuned x+z component R² values: most modes are near-perfect, but the last mode is much weaker.</div>

</div>
<div class="card small">

### Tuned x+z component fit

- <code>v1</code> through <code>v7</code>: **1.0000**
- <code>v8</code>: **0.9991**
- <code>v9</code>: **0.9989**
- <code>v10</code>: **0.9992**
- <code>v11</code>: **0.9986**
- <code>v12</code>: **0.3957**

### Observable-selection winners from the smaller objective table

| Objective | Winner |
| --- | --- |
| Tight alignment | z only |
| Balanced | x and z |
| Early warning | x only |

### Interpretation

- Objective choice matters even before the full channel-combo benchmark.
- The weakest mode is the main bottleneck in the tuned x+z model.

</div>
</div>

---

<!-- _class: dark -->

# Final mHAVOK Takeaways

<div class="cols">
<div class="card">

### What is now in the code

- Standalone manual mHAVOK implementation in <code>mhavok_lorenz.py</code>
- Exported CSV summaries and slide-ready figures in <code>plots/mhavok_lorenz</code>
- Dedicated metrics for **recall, precision, accuracy, R², and Chamfer distance**

</div>
<div class="card">

### Scientific conclusion

- The best mHAVOK model depends on the objective.
- For the full channel benchmark, **x+y+z** is the best event-detection model on the tested grid.
- For the default x+z setup, **best alignment** and **best event F1** are different configurations.

</div>
</div>

<div class="kpi">
<div class="box"><span class="label">Best x+z delays, rank</span><span class="value">150, 11</span></div>
<div class="box"><span class="label">Best combo delays, rank</span><span class="value">150, 5</span></div>
<div class="box"><span class="label">Best combo Chamfer</span><span class="value">0.149 s</span></div>
</div>