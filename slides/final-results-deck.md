# Final Results Deck Outline

## Slide 1. Research Question

Primary message:
- Hindmarsh-Rose: does HAVOK forcing turn on before burst onset strongly enough to be useful as an early-warning signal?
- Lorenz: how much do mHAVOK conclusions depend on rank, delays, and observable choice?
- Main headline: HAVOK is strong as an onset-warning signal in HR, while mHAVOK model ranking in Lorenz depends strongly on the objective and the observable set.

Suggested visual:
- Title-only slide, or a two-column layout with a simple HR burst trace on the left and a Lorenz attractor snapshot on the right.

Talking points:
- Frame the project as two related questions: event warning in a bursting neural model, and model-selection sensitivity in a canonical chaotic system.
- State upfront that the HR and Lorenz sections answer different things: HR is about predictive warning of bursts, while Lorenz is about what changes when the scoring rule changes.

## Slide 2. Hindmarsh-Rose Bursting Regime

Primary message:
- The simulated HR system is a sustained bursting regime with enough events to test warning performance quantitatively.

Suggested visuals:
- ../plots/havok_hr_attractor.png
- ../plots/havok_hr_burst_detection.png

Key numbers:
- Total simulated time: 49049.9 time units
- Bursts detected: 216
- Mean IBI: 227.8
- IBI standard deviation: 81.4
- Coefficient of variation: 0.36

Talking points:
- Emphasize that the prediction target is burst onset, not generic high-amplitude activity.
- Use the attractor plot to remind the audience that the underlying regime is low-dimensional enough to motivate a HAVOK-style embedding, but still irregular enough that onset timing is nontrivial.

## Slide 3. HR Event-Warning Performance

Primary message:
- The HAVOK forcing threshold gives very strong burst-onset recall over practical lead windows.

Suggested visuals:
- ../plots/havok_hr_recall_vs_window.png
- ../plots/havok_hr_lead_times.png
- ../plots/havok_hr_burst_detection.png

Key numbers:
- Forcing threshold: 0.001931
- Active fraction: 0.203
- Predicted burst onsets: 309
- Recall at 50, 30, 20, and 10 time units: 1.000
- Precision at 50, 30, 20, and 10 time units: 0.699
- Mean lead time: 8.0 time units

Talking points:
- The main positive result is not that every threshold crossing is clean, but that every true burst has a preceding forcing activation over a wide range of lead windows.
- The precision value of 0.699 means the warning signal is not perfectly selective, but it is much better than a naive baseline and still practically useful as a screening signal.
- Mention that the 5-time-unit precision collapse reflects a timing-resolution issue near onset rather than a complete failure of the forcing signal.

## Slide 4. HR Sensitivity to Rank, Delays, and Input Channel

Primary message:
- The warning result is not isolated to one fragile setting, but the best-performing configuration sits at high delays and rank.

Suggested visuals:
- ../plots/havok_hr_rank_delay_sensitivity.png
- ../plots/havok_hr_channel_comparison.png
- ../plots/havok_hr_singular_spectrum.png

Key numbers:
- Best x-channel setting at a 20-time-unit window: delays = 150, rank = 15
- Best x-channel metrics: recall = 1.000, precision = 0.867, mean linear R² = 0.946
- Best y-channel metrics: recall = 1.000, precision = 0.867, F1 = 0.929, mean linear R² = 0.938
- Best z-channel metrics: recall = 1.000, precision = 0.848, F1 = 0.918, mean linear R² = 0.972

Talking points:
- The best onset-warning performance appears at the largest tested delay stack, which suggests that temporal context matters more than aggressive rank truncation.
- x and y are effectively tied on event-detection quality, while z fits the linear latent dynamics slightly better but gives slightly worse event precision.
- That split is worth stating directly: the best state-fit channel is not exactly the best warning channel.

## Slide 5. HR Caveat Slide: Warning Success Is Not Full-State Success

Primary message:
- HR onset warning is strong, but the current HAVOK model is not a faithful full-state generative surrogate.

Suggested visuals:
- ../plots/havok_hr_reconstruction_quality.png
- ../plots/havok_hr_ibi_distribution.png
- ../plots/havok_hr_extremes.png

Key numbers:
- Forcing-mode R²: 0.5128
- Train reconstruction R²_rec: -1935.3039
- Test reconstruction R²_rec: -518.8316
- True IBI mean: 227.8 versus predicted IBI mean: 159.0
- KS statistic for IBI distributions: 0.3191 with p-value 0.0000
- True exceedances above 1.5: 6360
- Reconstructed exceedances above 1.5: 145401

Talking points:
- State this explicitly: the model is useful as an event-warning detector, but not yet reliable as a free-running surrogate for burst statistics or tail-risk analysis.
- The forcing coordinate is exactly where the nonlinearity is concentrated, so weak fit there can coexist with good onset signaling and poor long-horizon rollout.
- This slide protects the main claim by narrowing it to the part the current results actually support.

## Slide 6. Lorenz Rank-Delay Sweep Depends on the Objective

Primary message:
- In the Lorenz notebook, the “best” model changes depending on whether the objective is alignment or event detection.

Suggested visuals:
- Use the six-panel sensitivity heatmap from the current output of the rank-delay sweep cell in mHAVOK_lorenz.ipynb.
- Pair it with the exported table plots/mhavok_lorenz/rank_delay_event_metrics.csv if you want a compact text-only version.

Key numbers for the default x+z setup:
- Best event-F1 setting: delays = 150, rank = 11
- Event-F1 metrics at that setting: recall = 0.6429, precision = 0.2535, accuracy = 0.8801, F1 = 0.3636, Chamfer distance = 0.2352 s, mean linear R² = 0.9382
- Best tight-alignment setting: delays = 150, rank = 13
- Tight-alignment metrics at that setting: median |gap| = 0.0275 s, recall = 0.6071, precision = 0.2208, F1 = 0.3238, mean linear R² = 0.9493

Talking points:
- This is the core Lorenz lesson: “best model” is not a property of the method alone; it is a property of the scoring rule.
- The notebook already encodes this split by sorting alignment and event metrics separately.
- Make clear that the baseline-versus-tuned comparison figure is showing the best alignment model, not the best event-F1 model.

## Slide 7. Lorenz Observable Choice Changes the Ranking

Primary message:
- Observable choice changes the ranking materially, and adding y improves event detection in the tested grid.

Suggested visuals:
- Use the all-channel-combination comparison figure from the current output of the full combo benchmark cell in mHAVOK_lorenz.ipynb.
- Support with the exported table plots/mhavok_lorenz/all_channel_combo_metrics.csv.

Key numbers:
- Best overall event-detection combo on the tested grid: x+y+z with delays = 150 and rank = 5
- Metrics for x+y+z: recall = 0.7143, precision = 0.4878, accuracy = 0.8924, F1 = 0.5797, mean linear R² = 0.7450, Chamfer distance = 0.1488 s, median |gap| = 0.0005 s
- x+y: F1 = 0.4337
- x only: F1 = 0.4286
- x+z: F1 = 0.3636

Talking points:
- The full combination benchmark is stronger than the earlier x-only, z-only, and x+z comparison because it shows that the omitted y channel contains event-relevant information.
- The best three-channel event model has weaker mean linear R² than several simpler channel sets, which again reinforces the objective tradeoff.

## Slide 8. Lorenz Diagnostics: Linear Modes Fit Well, Last Mode Does Not

Primary message:
- The tuned Lorenz model fits most linear components almost perfectly, but the last component remains the bottleneck.

Suggested visuals:
- Use the singular-spectrum, operator-heatmap, and forcing-histogram figure from the Lorenz diagnostics cell.
- Support with plots/mhavok_lorenz/tuned_component_r2.csv.

Key numbers:
- Component R² values: v1 through v7 = 1.0000, v8 = 0.9991, v9 = 0.9989, v10 = 0.9992, v11 = 0.9986, v12 = 0.3957

Talking points:
- This is the Lorenz analog of the HR forcing issue: the final coordinate is where the simple linear-plus-input picture is weakest.
- That does not erase the usefulness of the decomposition, but it explains why model ranking depends on the evaluation target.

## Slide 9. Final Takeaways

Primary message:
- HAVOK forcing is a strong early-warning signal for HR burst onset, but not yet a reliable generative model of HR burst statistics.
- In Lorenz mHAVOK, rank, delay, and observable choice all matter, and the best model depends on whether you care about tight alignment, balanced performance, or event detection.

Suggested visual:
- A clean summary slide with two columns: HR takeaways on the left and Lorenz takeaways on the right.

Talking points:
- HR conclusion: the forcing coordinate is informative for warning, even when free simulation quality is weak.
- Lorenz conclusion: model selection has to be objective-first; there is no single universal winner.
- End with the honest claim: the main contribution here is not just a set of plots, but a clarified distinction between warning quality and generative fidelity.