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
    font-size: 2.05em;
    letter-spacing: 0.02em;
  }

  h2 {
    font-size: 1.42em;
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

  .grid-4 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 14px;
  }

  .card {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(20, 54, 66, 0.12);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 14px 32px rgba(20, 54, 66, 0.08);
  }

  .lead .card,
  .dark .card {
    background: rgba(255, 247, 232, 0.11);
    border: 1px solid rgba(255, 247, 232, 0.16);
    box-shadow: none;
  }

  .kpi {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 12px;
  }

  .kpi .box {
    background: rgba(20, 54, 66, 0.08);
    border-radius: 14px;
    padding: 10px 12px;
  }

  .lead .kpi .box,
  .dark .kpi .box {
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

  .lead .caption,
  .dark .caption {
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

  .lead th,
  .dark th {
    color: #fff7e8;
    border-bottom: 2px solid rgba(255, 247, 232, 0.28);
  }

  .lead td,
  .dark td {
    border-bottom: 1px solid rgba(255, 247, 232, 0.14);
  }

  .small {
    font-size: 0.74em;
  }

  .tight li {
    margin: 0.2em 0;
  }
---

<!-- _class: lead -->

# Hindmarsh-Rose mHAVOK Final Results
## 80k-sample burst-warning sweep

<div class="cols">
<div class="card">

### Question

- Can a multichannel HAVOK embedding produce a forcing-like signal that warns us before HR burst onset?
- Which observable set actually carries the cleanest warning information?

</div>
<div class="card">

### Bottom line

- **Observable choice matters more than channel count.**
- On the tested grid, **z only** is the best warning model.
- The full-state **x+y+z** model is useful, but it is **not the winner**.

</div>
</div>

<div class="kpi">
<div class="box"><span class="label">Post-transient samples</span><span class="value">80,000</span></div>
<div class="box"><span class="label">Burst onsets</span><span class="value">39</span></div>
<div class="box"><span class="label">Best warning combo</span><span class="value">z only</span></div>
</div>

---

# Prediction Target

<div class="cols-3070">
<div class="card tight">

### What we want to predict

- The target is **burst onset**, not generic large-amplitude activity.
- A useful warning signal must turn on **before** the burst begins.
- Single-channel HAVOK already shows that forcing activation clusters near burst onset.
- The mHAVOK question is whether **multiple observables** sharpen that warning or reveal which variables matter most.

### Why HR is a good test

- The regime is irregular enough that burst timing is nontrivial.
- It is still structured enough that a delay-embedding approach is scientifically reasonable.

</div>
<div>

![w:1000](../plots/havok_hr_burst_detection.png)

<div class="caption">Reference HR burst-detection plot: the presentation target is the onset of each burst, and the forcing-like signal is useful only if it activates beforehand.</div>

</div>
</div>

---

# Single-Channel HAVOK Baseline

<div class="cols">
<div>

![w:1000](../plots/havok_hr_recall_vs_window.png)

<div class="caption">The original HAVOK warning signal already captures every true burst across 50, 30, 20, and 10 time-unit windows.</div>

</div>
<div>

![w:1000](../plots/havok_hr_lead_times.png)

<div class="caption">Those activations are not just present; they arrive before onset, with a mean lead time of about 8.0 time units.</div>

</div>
</div>

<div class="card tight" style="margin-top: 18px;">

### Why this matters for the final results

- This establishes the baseline scientific fact: the forcing idea is already useful as a warning signal in HR.
- The mHAVOK sweep is therefore not asking whether warning is possible at all.
- It is asking **which observables carry that warning information most cleanly**.

</div>

---

# What mHAVOK Does

<div class="grid-4">
<div class="card tight">

### 1. Observe channels

- Start from the measured HR coordinates: **x(t), y(t), z(t)**.
- The goal is to ask which of these channels carries burst-warning information.

</div>
<div class="card tight">

### 2. Build stacked Hankel blocks

- Form one delay-embedding Hankel matrix per channel.
- Stack those blocks vertically so the SVD sees **time history across channels**.

</div>
<div class="card tight">

### 3. Extract latent coordinates

- Use the SVD to find low-dimensional coordinates.
- Treat the last retained coordinate as the **forcing-like mode**.

</div>
<div class="card tight">

### 4. Convert forcing into warnings

- Threshold the forcing signal.
- Merge nearby activations into one event.
- Score whether each predicted event lands inside a pre-burst warning window.

</div>
</div>

<div class="kpi">
<div class="box"><span class="label">Warning window</span><span class="value">20.0 tu</span></div>
<div class="box"><span class="label">Forcing quantile</span><span class="value">0.80</span></div>
<div class="box"><span class="label">Merge gap</span><span class="value">30.0 tu</span></div>
</div>

---

# Full Sweep Setup

<div class="cols">
<div class="card tight">

### Sweep design

- Post-transient analysis window: **80,000** samples
- Detected burst onsets: **39**
- Delay grid: **50, 100, 150**
- Rank grid: **5, 7, 9, 11, 13**
- Observable sets: **x, y, z, x+y, x+z, y+z, x+y+z**

</div>
<div class="card tight">

### What the sweep is testing

- Does adding channels improve burst-warning quality?
- Does the full-state model **x+y+z** beat simpler observable sets?
- Is the scientifically useful result “more channels help,” or is it “the right channel matters most”?

</div>
</div>

<div class="card tight" style="margin-top: 18px;">

### Evaluation rule

- A model is good when it produces **high recall and precision** for warning events before burst onset.
- We also track lead time and alignment, but the practical question is still: **does the warning arrive early enough and selectively enough to be useful?**

</div>

---

# Main Results: Observable Choice Dominates

![w:1230](hr-mhavok-fullrun-comparison.png)

<div class="caption">Exact 80k-sample final-result comparison: the full-state x+y+z model is useful, but z only is better on recall, precision, F1, and median alignment gap.</div>

<div class="cols" style="margin-top: 16px;">
<div class="card tight">

### Strongest result

- The best warning model on the tested grid is **z only**, not the full-state model.
- Its median absolute gap is effectively **0.0** time units in the completed full-run summary.

</div>
<div class="card tight">

### Interpretation

- The slow adaptation variable **z** appears to carry the cleanest burst-cycle information.
- Adding every channel does **not** automatically improve warning quality.

</div>
</div>

---

<!-- _class: dark -->

# What We Can Claim Safely

<div class="cols">
<div class="card tight">

### Defensible claim

- HR mHAVOK is useful as an **observable-screening tool**.
- It shows that burst-warning quality depends strongly on **which channels are included**.
- On this grid, **z** and not **x+y+z** is the best warning input.

</div>
<div class="card tight">

### Important limit

- Do **not** oversell this as “mHAVOK beats HAVOK everywhere.”
- The strongest direct warning result in the repo is still the original single-channel HAVOK story.
- The multichannel result is valuable because it clarifies **which observables carry warning information**.

</div>
</div>

<div class="kpi">
<div class="box"><span class="label">Best z-only F1</span><span class="value">0.729</span></div>
<div class="box"><span class="label">Best x+y+z F1</span><span class="value">0.514</span></div>
<div class="box"><span class="label">Best x-only F1</span><span class="value">0.000</span></div>
</div>

---

# Presentation Takeaways

<div class="cols">
<div class="card tight">

### How to present this

- Lead with the **prediction problem**: warn before burst onset.
- Spend time explaining what the forcing-like coordinate means.
- Show one onset plot and two summary graphs; keep the dense sweep behind GitHub or a QR code.

</div>
<div class="card tight">

### Scientific takeaway

- The useful HR mHAVOK result is **not** “more channels are better.”
- The useful result is that **channel choice is a first-order modeling decision**.
- For the tested 80k-sample sweep, the cleanest warning signal comes from **z**.

</div>
</div>

<div class="card tight" style="margin-top: 18px;">

### One-sentence conclusion

- In Hindmarsh-Rose, the forcing-based warning idea remains strong, and the new mHAVOK sweep shows that **the slow adaptation variable is the most informative warning observable on the tested grid**.

</div>