# mHAVOK Results Talking Points

## Slide 1. mHAVOK on Lorenz

- mHAVOK, or multivariate HAVOK, replaces the standard single-channel delay embedding with a stack of Hankel matrices built from multiple observables at once.
- The goal is to find a low-dimensional latent coordinate system where most modes evolve linearly and one forcing-like mode captures the irregular switching behavior.
- This presentation has two main results.
- First, the best model depends on which objective is used to score it.
- Second, the choice of Lorenz channels changes the event-detection ranking materially.
- The headline numbers are that the best default `x+z` event F1 is 0.364, while the best all-channel model `x+y+z` reaches 0.580.
- That improvement shows that the `y` channel carries independent information that matters for event detection.

## Slide 2. Step 1 — Building the Hankel Matrix

- The Hankel matrix is the foundation of every HAVOK-style method.
- For a single channel it has `delays` rows and roughly `n_timesteps` columns.
- In mHAVOK, one Hankel matrix is built for each observable and then stacked vertically.
- That means the full matrix has `n_channels × delays` rows.
- This lets the SVD identify latent directions that jointly span the delay history of all observed channels.
- In the actual code, `build_hankel(signal, delays)` is just a list comprehension that slices the same signal many times with a one-step offset.
- The first row is the original signal segment, the second row is shifted by one time step, and so on until the full delay window is filled.
- That structure is what converts a one-dimensional time series into a snapshot of recent history.
- The call `np.vstack([build_hankel(channel, delays) for channel in Y])` is the multivariate step.
- `Y` contains the chosen observables, so this one line is where the method changes from ordinary HAVOK to mHAVOK.
- If `Y` contains `x` and `z`, the final matrix stacks the delay history of `x` directly on top of the delay history of `z`.
- If `Y` contains `x+y+z`, the matrix becomes taller, and the SVD has access to more information about the trajectory.
- The parameter `delays` controls how much temporal context is encoded.
- A larger delay window captures slower and longer-timescale dynamics, but also increases the size of the matrix.
- In the rank-delay sweep, larger delays consistently improved event detection.
- That suggests the Lorenz lobe-switching behavior leaves a temporal footprint that spans at least 150 time steps.
- The economy SVD with `full_matrices=False` is important because only the leading latent coordinates are needed, not the full unitary basis.
- In the code, the right singular vectors `Vh` are the key object because they are indexed by time and become the latent coordinate trajectories.
- The line `V = Vh[:rank, :].T` keeps only the first `rank` singular directions and transposes them so time runs down the rows.
- That means each row of `V` is one time point in reduced coordinates, and each column is one latent mode.
- This orientation is important because the next regression step treats the columns of `V` as state variables evolving over time.
- After the SVD, the first `rank - 1` coordinates are treated as the linear state, and the last coordinate becomes the forcing-like mode.

- One useful way to say this verbally is: the Hankel matrix creates memory, and the SVD compresses that memory into the few latent coordinates that explain the most structure.

## Slide 3. Step 2 — Regression for A and B

- Once the latent coordinates `V` are available, the next step is to fit a linear model for the time derivative of the linear coordinates.
- The regression uses the full state-plus-forcing vector `Theta` as the predictor and `dV/dt` as the response.
- The core modeling assumption is that the first `rank - 1` modes evolve approximately linearly.
- The last mode is intended to collect the more irregular, nonlinear content.
- In the code this split is made explicitly with `V_linear = V[:, : rank - 1]` and `forcing = V[:, rank - 1]`.
- So the model says: treat every latent mode except the last one as part of the reduced linear state, and treat the last mode as an external drive.
- The derivative `dVdt = np.gradient(V_linear, dt, axis=0)` is computed numerically from the latent trajectories.
- That derivative is the quantity the regression is trying to predict.
- The line `Theta = np.column_stack([V_linear, forcing])` builds the design matrix by appending the forcing column to the linear state columns.
- In other words, each row of `Theta` contains the reduced state and the forcing value at one time step.
- Fitting `LinearRegression(fit_intercept=False)` then estimates the best linear mapping from that augmented state to the derivative of the linear coordinates.
- This is the precise meaning of the HAVOK ansatz $\dot{v} = A v + B f(t)$ in code form.
- If this assumption is working, the linear modes should have very high componentwise $R^2$, while the last forcing-like coordinate will be noticeably weaker.
- That is exactly what appears later in the tuned `x+z` results, where most modes are near 1.0 and the last mode is much lower.
- `LinearRegression(fit_intercept=False)` is the right choice here because the SVD-centered coordinates already absorb the mean structure.
- Adding an intercept would distort the forcing coordinate and contaminate the event-thresholding step.
- The fitted matrix `A` is the reduced linear operator, and `B` tells us how strongly the forcing coordinate drives each state mode.
- In the code, `Xi = model.coef_.T` is the combined coefficient matrix.
- The first `rank - 1` rows of `Xi` become `A`, and the last row becomes `B`.
- That slicing is important because it matches the model decomposition exactly: state-to-state coupling lives in `A`, while forcing-to-state coupling lives in `B`.
- The script also computes `dVdt_pred = Theta @ Xi` and compares that prediction against the true `dVdt` to get per-component $R^2$ values.
- So mean linear $R^2$ is not coming from the original Lorenz variables directly.
- It is measuring how well the reduced latent dynamics are explained by the linear-plus-forcing model.
- The average of the per-component regression $R^2$ values is what is reported throughout as mean linear $R^2$.

- A simple way to explain the code is: first estimate the latent derivatives, then learn the best linear rule that predicts those derivatives from the current latent state plus the forcing coordinate.

## Slide 4. x+z Baseline vs Tuned mHAVOK Models

- The baseline model uses `delays = 100` and `rank = 9`, which came from the original notebook default.
- That baseline performs poorly for event detection, with recall 0.1786 and F1 0.1099.
- So the original default should not be treated as close to optimal for this task.
- Both tuned models use `delays = 150`, but they choose different ranks depending on the scoring rule.
- The best alignment model uses rank 13.
- The best event-F1 model uses rank 11.
- That small rank difference is enough to change which objective wins.
- The alignment winner has the tighter temporal gap metric.
- The event winner has better recall and better precision.
- All three models are trained on the same observable set and the same underlying trajectory.
- The only thing that changes is which hyperparameter setting is selected.
- This slide is meant to make one point clear: model ranking is driven by the evaluation objective, not just by the method itself.

## Slide 5. Step 3 — Thresholding and Event Detection

- The forcing coordinate is a continuous time series, so it must be converted into discrete events before scoring.
- That is done by thresholding the magnitude of the forcing signal at the 95th percentile.
- Any time the absolute forcing exceeds that threshold, the model declares the system to be active.
- This makes the active set sparse, at about 5 percent of all time points.
- In code, `forcing_threshold = np.quantile(np.abs(forcing), 0.95)` means the threshold is not hand-picked.
- It is determined directly from the empirical distribution of the forcing magnitude.
- Using the absolute value makes positive and negative forcing bursts count equally as indicators of a possible switch.
- The mask `active_mask = np.abs(forcing) >= forcing_threshold` is therefore a boolean time series marking the strongest forcing excursions.
- Predicted event times are then defined as the rising edges of that active mask.
- In other words, each transition from inactive to active becomes one predicted lobe-switch time.
- The line `np.diff(active_mask.astype(int)) == 1` is a compact way to detect exactly those inactive-to-active transitions.
- Converting to integers makes `False -> 0` and `True -> 1`, so a rising edge appears as a jump of `+1` in the discrete difference.
- Adding 1 to those indices moves the event marker back to the actual onset time instead of the time step before it.
- The next line `event_times = time_havok[onset_indices]` converts those onset indices into physical times that can be compared against the true switch times.
- Those predicted event times are matched against the true Lorenz lobe-switch times using a tolerance of 0.10 seconds.
- That tolerance is wide enough to absorb small timing jitter, but narrow enough that a clearly late prediction still counts as a miss.
- In the underlying implementation, `match_event_times` performs a forward scan through the true and predicted event lists.
- It is a one-to-one greedy matching procedure, so one predicted event cannot be used to explain multiple true switches.
- That avoids artificially inflating recall when the forcing signal flickers several times near a single switch.
- Recall and precision are then computed from the matched and unmatched events.
- F1 summarizes the tradeoff between the two.
- The code stores the number of matched events as `tp`, then computes recall as a fraction of true switches recovered and precision as a fraction of predicted events that were correct.
- The F1 score is then the harmonic mean of recall and precision, so it only becomes large if both are reasonably strong.
- Accuracy is different because it is computed over every time step, not just over event times.
- That is why accuracy can remain high even when recall is relatively poor.
- For example, the baseline still has accuracy above 0.86 even though its recall is only about 0.18.
- The line `true_labels = compute_temporal_labels(...)` creates a dense time-series label around each true event using the same tolerance window.
- Comparing those dense labels to `active_mask` gives a time-point-wise accuracy score.
- That metric asks a different question from recall and precision: not "did we hit the switch times?" but "how often is the forcing activity label right at each time step?"

- A good verbal summary here is: thresholding turns a continuous forcing signal into a sparse event detector, and the rest of the code measures how well those detected onsets line up with the true Lorenz switches.

## Slide 6. Rank-Delay Sweep for Event Metrics

- The sweep covers delays in `{50, 100, 150}` and ranks in `{5, 7, 9, 11, 13, 15}`.
- Every cell in the grid is a separate mHAVOK fit and evaluation.
- The strongest trend is that recall, F1, and Chamfer distance all improve as delays increase from 50 to 150.
- That means the lobe-switching dynamics require a fairly long temporal context window to be separated well in latent space.
- Mean linear $R^2$ stays high over much of the grid.
- That makes it a useful fit metric, but not a sufficient event-detection metric.
- The grid cell with the best mean linear $R^2$ is not the same as the one with the best F1.
- The table on the slide shows this explicitly.
- If the objective is tight temporal alignment, the best setting is `delays = 150`, `rank = 13`.
- If the objective is best event F1, the best setting is `delays = 150`, `rank = 11`.
- The main lesson is that the sweep is easy to run, but the winning model depends entirely on what “best” means.

## Slide 7. Step 4 — Chamfer Distance

- Chamfer distance comes from point-cloud comparison, but it applies naturally to one-dimensional event sets as well.
- It is defined symmetrically.
- First, each true lobe-switch time is compared to the nearest predicted event.
- Then, each predicted event is compared to the nearest true lobe-switch.
- The average of those two nearest-neighbor distances is the Chamfer distance.
- In code, the first step is building the matrix `D = np.abs(ref[:, None] - cand[None, :])`.
- That matrix contains every pairwise absolute time difference between true events and predicted events.
- Each row corresponds to one reference event and each column corresponds to one candidate event.
- Taking `D.min(axis=1)` gives the distance from each true switch to its nearest prediction.
- Taking `D.min(axis=0)` gives the distance from each prediction to its nearest true switch.
- Averaging those two vectors and multiplying by 0.5 makes the score symmetric.
- That symmetry matters because it penalizes both missed events and spurious extra detections.
- The advantage over F1 is that Chamfer distance does not depend on a hard tolerance threshold.
- F1 treats a near miss and a very large miss the same way once they fall outside the tolerance window.
- Chamfer distance instead degrades continuously as timing gets worse.
- That makes it a useful complement to F1 rather than a replacement.
- The helper function also explicitly returns `inf` if either event set is empty.
- That is a deliberate safeguard: if a model predicts no events, or if there are no reference events, the distance should be treated as maximally bad rather than silently producing a misleading finite number.
- In the channel-combo comparison, `x+y+z` gives the lowest Chamfer distance at 0.1488 seconds.
- That means predicted events and true events are, on average, within about 0.15 seconds of a partner.
- One limitation is that Chamfer distance can become large when there are many spurious predictions far away from any true event.
- That is why it should always be interpreted alongside precision.

- A concise way to explain the code is: Chamfer distance constructs the full event-to-event distance table, then asks how close each event is to its nearest counterpart in either direction.

## Slide 8. Channel Combinations: x, y, z Matter

- Each row in this comparison uses the best model found for that channel combination over the full rank-delay grid.
- The best event-detection model overall is `x+y+z` with `delays = 150` and `rank = 5`.
- It achieves recall 0.7143, precision 0.4878, F1 0.5797, and Chamfer distance 0.1488 seconds.
- Adding `y` changes the ranking substantially.
- That means `y` contains event-relevant information that is not fully recoverable from `x` and `z` alone.
- This is plausible dynamically because `y` is tightly coupled to `x` in the Lorenz equations and carries rapid oscillatory information.
- The `x+y+z` winner also uses the lowest optimal rank among the top-performing combinations.
- That suggests a richer observable set lets the SVD capture the important structure with fewer retained latent modes.
- Its mean linear $R^2$ is lower than simpler channel sets such as `x` alone or `x+y`.
- That is expected because a rank-5 model on a three-channel Hankel stack is more compressed.
- The residual variance is pushed into the forcing coordinate, which is precisely what helps event detection.
- The practical takeaway is straightforward.
- If all three channels are available, use all three.
- If only one or two are available, `x` and `x+y` are the strongest fallbacks in the tested grid.

## Slide 9. R² Diagnostics and Objective Tradeoffs

- The componentwise $R^2$ plot shows the regression quality mode by mode for the tuned `x+z` model.
- Modes `v1` through `v7` are essentially perfect to four decimal places.
- Modes `v8` through `v11` are still above 0.998.
- That means the dominant latent structure is captured extremely well by the linear-plus-forcing decomposition.
- The final mode, `v12`, is much weaker at 0.3957.
- That last mode is the forcing coordinate.
- Its low $R^2$ means the regression cannot linearly predict its derivative nearly as well as it can for the other modes.
- That is not a bug in the method.
- It is the expected bottleneck, since the forcing coordinate is deliberately meant to absorb irregular or nonlinear behavior.
- The smaller objective-selection table on the right reinforces the same pattern seen elsewhere.
- `z` alone is best for tight alignment.
- `x` alone is best for early warning.
- `x+z` is the balanced compromise.
- So even before the full channel-combo benchmark, the winner already depends on the scoring rule.
- The main interpretation is that the linear latent dynamics are fit very well, while the forcing mode remains the weakest and most informative coordinate.

## Slide 10. Final mHAVOK Takeaways

- There are four main implementation steps behind the presentation.
- First, build a stacked multi-channel Hankel matrix and compress it with SVD.
- Second, fit a reduced linear model of the form $dV/dt = A V_{linear} + B f(t)$.
- Third, threshold the forcing magnitude, extract event onset times, and compute recall, precision, F1, accuracy, and related event metrics.
- Fourth, compute Chamfer distance as a continuous timing-error metric.
- The scientific conclusion is that mHAVOK is sensitive to three major choices: delays, rank, and observable set.
- The best alignment model and the best event-detection model are not the same.
- Over the full sweep, the best overall event-detection model is `x+y+z` with `delays = 150` and `rank = 5`.
- That model reaches F1 0.580 and Chamfer distance 0.149 seconds.
- The pipeline is fully reproducible from the standalone script.
- Running `python mhavok_lorenz.py` regenerates the figures and CSV summaries used throughout the presentation.