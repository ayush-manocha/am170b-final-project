# Figure Scripts

Use this as a short speaking script for each PNG in this folder.

Quick context:
- `hr-mhavok-fullrun-comparison.png` uses the confirmed **80k-sample full-run** result.
- The `mhavok_hr_*.png` figures in this folder come from the packaged **30k export**.
- So use the full-run figure for the final headline claim, and the packaged figures as supporting diagnostics.

## hr-mhavok-fullrun-comparison.png

### What it represents

- This is the final summary figure for the HR mHAVOK result.
- It compares three observable choices: `x only`, `x+y+z`, and `z only`.
- The metrics are all warning metrics: recall, precision, F1, mean lead time, and median timing gap.

### Concise script

- “This is the main final-result figure from the 80k-sample run.”
- “It shows that `z only` is the best warning input on the tested grid.”
- “The full-state `x+y+z` model is still useful, but it is weaker than `z only` on recall, precision, F1, and timing alignment.”
- “The `x`-only model fails completely here, so observable choice is clearly not a minor detail.”

### What it means

- We are not just fitting latent dynamics well; we are asking which observable gives the cleanest warning before burst onset.
- This figure says the slow adaptation variable `z` carries the most useful warning information on the tested grid.
- The key scientific message is: **more channels are not automatically better; the right channel matters most**.

### Numbers to cite

- `z only`: recall `1.0000`, precision `0.5735`, F1 `0.7290`, mean lead `0.1308`, median `|gap| = 0.0000`.
- `x+y+z`: recall `0.7179`, precision `0.4000`, F1 `0.5140`, mean lead `0.1710`, median `|gap| = 0.1000`.
- `x only`: recall `0.0000`, precision `0.0000`, F1 `0.0000`.



## mhavok_hr_x_only_model_comparison.png

### What it represents

- This is the simplest baseline-versus-tuned figure in the packaged 30k export.
- It compares the best `x`-only configuration with the tuned `x+y+z` configuration.
- It is useful because it shows, very directly, what multichannel information can buy you over a weak single-channel baseline.

### Concise script

- “This figure compares a weak `x`-only baseline with the tuned full-state model in the packaged sweep.”
- “The tuned `x+y+z` model is much better on recall, precision, and F1.”
- “So this figure motivates why it is worth trying multichannel HAVOK at all.”

### What it means

- `x` alone does not capture the burst-warning structure well in this packaged sweep.
- Letting the model use the full state can recover a much cleaner warning signal.
- This is not yet the whole observable-choice story, but it is the easiest visual proof that channel choice matters.

### Numbers to cite

- Best `x only`: recall `0.2105`, precision `0.1481`, F1 `0.1739`, mean linear `R² = 0.8730`, mean lead `0.0500`.
- Best `x+y+z`: recall `0.7895`, precision `0.5556`, F1 `0.6522`, mean linear `R² = 0.9338`, mean lead `0.1533`.

### Caveat

- This figure is from the packaged 30k export, not the 80k full-run headline result.
- Use it as a supporting comparison, not as the final summary figure.

## mhavok_hr_channel_combo_metrics.png

### What it represents

- This figure compares the best-performing model from each observable family in the packaged 30k export.
- The main panels to focus on are recall, precision, and F1.
- The linear `R²` panel is secondary and should be read as a model diagnostic, not the primary task score.

### Concise script

- “This is the clearest observable-selection figure in the packaged sweep.”
- “The strongest models all contain `z`, while `x+y+z` is good but not the best.”
- “The `x+y` combination fails completely, which shows that adding channels can also make the warning structure worse.”

### What it means

- This figure is strong evidence that the slow variable `z` is central to burst warning.
- It also shows that observable selection is not a cosmetic choice; it changes the ranking materially.
- The important message is that the warning signal seems to depend on keeping the slow-timescale information carried by `z`.

### Numbers to cite

- `z`: recall `1.0000`, precision `0.7037`, F1 `0.8261`, median `|gap| = 0.0000`.
- `x+z`: recall `1.0000`, precision `0.7037`, F1 `0.8261`, median `|gap| = 0.0000`.
- `x+y+z`: recall `0.7895`, precision `0.5556`, F1 `0.6522`, median `|gap| = 0.2000`.
- `x`: recall `0.2105`, precision `0.1481`, F1 `0.1739`.
- `x+y`: recall `0.0000`, precision `0.0000`, F1 `0.0000`.

### Safe interpretation

- If asked “Does this mean `z` is all you need?”, say: for this warning objective and this tested grid, `z` is the dominant observable, but that does not mean the other variables are irrelevant for every task.

## mhavok_hr_rank_delay_metrics.png

### What it represents

- This is the parameter-sensitivity figure for the packaged 30k sweep.
- Each heatmap cell is a separate fitted model.
- The event panels show task performance, while the `R²` panel shows latent fit quality.

### Concise script

- “This figure shows that the method is sensitive to delay and rank, not just observable set.”
- “In the packaged sweep, the useful region is concentrated at `delays = 50`, especially around `rank = 5` and `rank = 9`.”
- “Some settings still have decent latent fit but zero event performance, so linear fit alone is not enough.”

### What it means

- This figure tells the audience not to trust one hand-picked hyperparameter setting too much.
- Warning quality only appears in a relatively narrow part of the tested parameter space.
- That means the event structure is being isolated only when the embedding has the right temporal depth and latent dimension.

### Numbers to cite

- Strong packaged cell: `delays = 50`, `rank = 9`, recall `0.7895`, precision `0.5556`, F1 `0.6522`, median `|gap| = 0.2000`.
- Another strong packaged cell: `delays = 50`, `rank = 5`, recall `0.7895`, precision `0.5556`, F1 `0.6522`, but with worse timing structure: mean lead `1.98`, median `|gap| = 1.3`.
- Failing cells include `delays = 50`, `rank = 7` and both `delays = 100` settings.

### Subtle point to mention

- `rank = 5` and `rank = 9` tie on packaged F1, but not on timing alignment.
- So a single metric does not tell the whole story.

## mhavok_hr_component_r2.png

### What it represents

- This is a model-diagnostic figure, not a warning-performance figure.
- It shows how well each latent coordinate is fit by the reduced linear-plus-forcing model.
- High `R²` means that coordinate is well captured; lower `R²` means that coordinate is harder to model.

### Concise script

- “Most latent coordinates are fit well, but the last one is noticeably weaker.”
- “That matches the usual HAVOK picture: the forcing-like direction is the hardest part to model linearly.”
- “This helps explain why a model can look decent internally and still vary a lot on event-detection performance.”

### What it means

- The decomposition is mostly coherent; these are not random latent directions.
- But the weakest component still concentrates the hardest unresolved behavior.
- So this figure supports a nuanced claim: the reduced model captures a lot of the latent structure, but not all of it equally well.

### Numbers to cite

- `v1 = 0.9392`
- `v2 = 0.9476`
- `v3 = 0.9785`
- `v4 = 0.9516`
- `v5 = 0.9785`
- `v6 = 0.9361`
- `v7 = 0.9532`
- `v8 = 0.7860`

### Safe interpretation

- Do not say `v8` is “bad.”
- Say it is the weakest relative to the others, and that relative drop tells you where the model struggles most.