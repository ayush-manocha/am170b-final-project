---
marp: true
paginate: true
size: 16:9
style: |
  section {
    background: linear-gradient(135deg, #f6f1e8 0%, #f9f7f2 55%, #e9f0ee 100%);
    color: #182430;
    font-family: "Avenir Next", "Aptos", "Segoe UI", sans-serif;
    padding: 46px 54px;
  }
  section.lead {
    background: radial-gradient(circle at top right, #d6e7e0 0%, #f8f3ea 52%, #efe4d2 100%);
  }
  h1, h2 {
    color: #15384f;
    margin-bottom: 0.25em;
  }
  p, li {
    font-size: 0.95em;
    line-height: 1.28;
  }
  strong {
    color: #0c5c78;
  }
  .columns {
    display: flex;
    gap: 28px;
    align-items: center;
  }
  .col {
    flex: 1;
  }
  .hero-metric {
    display: inline-block;
    margin-right: 18px;
    padding: 10px 14px;
    border-radius: 14px;
    background: rgba(21, 56, 79, 0.08);
    font-size: 0.82em;
  }
  .caption {
    font-size: 0.72em;
    color: #425466;
  }
  img {
    max-width: 100%;
    max-height: 520px;
    border-radius: 16px;
    box-shadow: 0 16px 36px rgba(21, 56, 79, 0.12);
  }
---

<!-- _class: lead -->

# Preliminary HAVOK Results
## Hindmarsh-Rose bursting dynamics from scalar observations

AM170B final project snapshot centered on the question: can HAVOK forcing predict burst onset in the Hindmarsh-Rose model?

<span class="hero-metric">Rank = 11</span>
<span class="hero-metric">Delays = 100</span>
<span class="hero-metric">100,000 training samples</span>

---

# Setup And Framing

<div class="columns">
<div class="col">

- The project replaces the Lorenz tutorial example with the Hindmarsh-Rose neuron model in a chaotic bursting regime.
- Only the scalar voltage-like variable $x(t)$ is supplied to HAVOK; the hidden goal is to recover useful low-dimensional dynamics from delayed coordinates.
- Current script settings use $\Delta t = 0.01$, rank 11, and 100 delay coordinates.
- The extended prediction section now runs after keeping the long forcing model on the same delay setting as the fitted predictor.

</div>
<div class="col">

![width:980px](./assets/overview.png)

<div class="caption">The full Hindmarsh-Rose attractor is generated for reference, but HAVOK is trained only on the scalar trace on the right.</div>

</div>
</div>

---

# Result 1: Reconstruction Is Stronger Than Forecasting

![width:1460px](./assets/reconstruction_prediction.png)

- Reconstruction of the observed training signal is reasonably faithful: correlation $\approx 0.872$ and relative RMSE $\approx 0.538$.
- When the model is driven by forcing from a longer run, prediction quality is visibly weaker but still structured: correlation $\approx 0.756$.
- Interpretation: the learned delayed linear model captures the local bursting geometry well, but longer-horizon phase accuracy still drifts.

---

# Observation / Interpretation

<div class="columns">
<div class="col">

![width:900px](./assets/forcing_alignment.png)

</div>
<div class="col">

- With the computed threshold, the forcing is active only about **9.7%** of the time.
- Every detected burst onset has a large-forcing event within **0.6 s**, with a median nearest-event gap of about **0.12 s**.
- The important caveat is timing: the nearest active forcing event occurs **before** onset for only about **36%** of bursts, so the signal is usually contemporaneous with onset rather than clearly predictive.
- Interpretation: in this preliminary run, HAVOK forcing behaves more like a transition marker for entry into bursting than a robust early-warning signal.
- Relative to the Lorenz paper framing, that still supports the core HAVOK idea that intermittency is concentrated in a sparse forcing coordinate, but the predictive lead time here looks weaker and still needs benchmarking.

</div>
</div>

---

# Result 3: The Embedding Looks Very Low Rank

<div class="columns">
<div class="col">

![width:900px](./assets/operator_spectrum.png)

</div>
<div class="col">

- The first five singular values already capture about **99.999%** of the Hankel energy; rank 11 is conservative rather than minimal.
- The learned operator is close to the expected skew-symmetric, banded structure, but not perfect: skewness residual $\approx 4.5\%$ and off-tridiagonal mass $\approx 12.2\%$.
- Interpretation: the delayed coordinates are strongly compressible, yet some non-ideal structure remains, which likely contributes to forecast drift.

</div>
</div>

---

# Expanded Preliminary Results

- From only the scalar signal $x(t)$, HAVOK recovers a compact delayed-coordinate model that reconstructs the training dynamics well: reconstruction correlation $\approx 0.872$ with relative RMSE $\approx 0.538$.
- The forcing channel is genuinely sparse, active only about **9.7%** of the time, which supports the idea that most of the bursting trajectory is handled by the near-linear delayed subsystem.
- Burst transitions are where the forcing becomes informative: every detected onset has a thresholded forcing event within **0.6 s**, and the median nearest-event gap is only **0.12 s**.
- Longer-run prediction is weaker than reconstruction but still nontrivial: prediction correlation $\approx 0.756$, which suggests the model retains qualitative burst structure even when phase accuracy starts to drift.
- The strongest preliminary claim is therefore not full burst-onset prediction yet, but that HAVOK successfully isolates the transition structure of the Hindmarsh-Rose system into a sparse forcing coordinate that stays tightly coupled to bursting events.