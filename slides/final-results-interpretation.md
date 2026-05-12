# Final Results Interpretation Notes

## Main Defensible Claims

### Hindmarsh-Rose

- The strongest defensible claim is that the HAVOK forcing signal is a useful early-warning marker for burst onset in the tested HR regime.
- The data support this because recall is 1.000 across 50, 30, 20, and 10 time-unit windows, with precision 0.699 over those same windows and a mean lead time of 8.0 time units.
- The stronger claim that HAVOK gives a faithful free-running surrogate for HR burst statistics is not supported by the current results.

### Lorenz

- The strongest defensible claim is that mHAVOK conclusions depend materially on the evaluation objective and the observable set.
- The rank-delay sweep shows that the best alignment model and the best event-F1 model are different even before changing observables.
- The all-channel benchmark shows that adding y changes the ranking materially, with x+y+z outperforming the earlier x-only, z-only, and x+z comparisons on event metrics.

## What To Say About the HR Caveats

### Short version for the presentation

- “The forcing coordinate is good at flagging upcoming burst onset, but the current free-running HAVOK reconstruction is not accurate enough to trust for full-state simulation or tail-statistics claims.”

### Why that statement is correct

- The forcing mode is the weakest-fit mode in the latent model, with R² = 0.5128.
- The free-running reconstruction is poor, with R²_rec = -1935.3039 on train and -518.8316 on test.
- The burst-interval and extreme-value summaries are mismatched: predicted IBI mean is 159.0 versus true mean 227.8, the KS test rejects equality of the IBI distributions, and reconstructed threshold exceedances are much too frequent.

### How to explain the apparent contradiction

- Event warning and free-running trajectory fidelity are not the same task.
- A thresholded forcing signal can still mark regime transitions reliably even if long-horizon simulation drifts badly.
- In other words, the model can be useful as a detector even when it is not yet useful as a surrogate generator.

### What not to overclaim

- Do not say the HR HAVOK model “reconstructs the bursting dynamics accurately” without qualification.
- Do not say the return-period or extreme-value outputs validate the model statistically; they currently show a mismatch, not a confirmation.
- Do not use the negative reconstruction R² values as a small technical footnote. They materially constrain the interpretation.

## What To Say About the Lorenz Objective Split

### Short version for the presentation

- “For Lorenz, the best mHAVOK model depends on what ‘best’ means. Tight alignment, event-F1, and full observable-set comparison do not choose the same model.”

### Concrete numbers to cite

- Default x+z setup, best event-F1 model: delays = 150, rank = 11, recall = 0.6429, precision = 0.2535, accuracy = 0.8801, F1 = 0.3636, Chamfer distance = 0.2352 s.
- Default x+z setup, best alignment model: delays = 150, rank = 13, median |gap| = 0.0275 s, F1 = 0.3238.
- Full combo benchmark winner: x+y+z with delays = 150 and rank = 5, recall = 0.7143, precision = 0.4878, accuracy = 0.8924, F1 = 0.5797, Chamfer distance = 0.1488 s.

### Important notebook-specific nuance

- The baseline-versus-tuned comparison cell in mHAVOK_lorenz.ipynb currently refits the best alignment model by taking sorted_results[0].
- That figure is therefore aligned with the median-gap objective, not with the event-F1 objective.
- If you are presenting event detection, cite the exported rank-delay table and the combo benchmark rather than implying that the baseline-versus-tuned overlay is the event-optimal configuration.

## How To Explain the Lorenz Component-R² Result

- Most tuned linear components are fit almost perfectly, but the last component remains much weaker with R² = 0.3957.
- That means the model captures the dominant linear latent structure well, but the final forcing-like coordinate is still where the reduced model struggles.
- This is why a model can look excellent on mean linear R² and still change ranking when evaluated by event metrics.

## Recommended One-Sentence Conclusions

### HR conclusion

- “In Hindmarsh-Rose, HAVOK forcing behaves like a strong burst-onset warning signal, but the current model should not yet be treated as a high-fidelity generative surrogate.”

### Lorenz conclusion

- “In Lorenz, mHAVOK model selection is objective-dependent: the best alignment model, the best event-detection model, and the best observable set are not the same.”

## Likely Questions and Safe Answers

### Why is HR recall perfect but reconstruction so poor?

- Because the onset-warning task only needs the forcing coordinate to activate near regime changes, while free-running reconstruction needs accurate long-horizon latent evolution.

### Does the HR extreme-value mismatch invalidate the warning result?

- No. It invalidates the stronger claim that the model captures burst-tail statistics, but it does not erase the separate result that forcing crosses threshold before true bursts.

### Which Lorenz model should we call “best”?

- Answer that by objective: z only is best for tight alignment among the original objective-selection table, x only is best for early warning in that same table, x+z is the balanced winner there, and x+y+z is the best event-detection model in the full combo benchmark.

### Why does adding y help?

- Because the full combo benchmark shows that y contributes event-relevant information that is not fully recoverable from x and z alone in the tested grid.

## Suggested Verbal Transition Between the Two Sections

- “The HR analysis answers whether forcing can warn about bursts. The Lorenz analysis answers a different but related question: once you have a HAVOK-style decomposition, how sensitive is your conclusion to the scoring rule and the observable choice?”