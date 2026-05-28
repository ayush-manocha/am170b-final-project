# Poster Reframe for HR mHAVOK

## Core Story

- Lead with the problem, not the sweep: we want a warning signal that turns on before a Hindmarsh-Rose burst begins.
- Explain mHAVOK as the mechanism for extracting that warning signal from multiple observables, then use only one or two figures to show that the forcing-like mode is informative.
- Keep the dense sweep outputs in the report or GitHub, not on the poster.

## Recommended Poster Boxes

### Box 1. Prediction target

- Show a simple HR burst trace and mark burst onsets.
- State the concrete question: can we predict burst onset from a forcing-like latent coordinate before the burst starts?
- Good visual: `../plots/havok_hr_burst_detection.png`

### Box 2. What mHAVOK does

- Use a clean 4-step graphic:
- Observe channels `x(t), y(t), z(t)`.
- Build one Hankel matrix per channel and stack them.
- Take an SVD to get latent coordinates.
- Fit linear dynamics to the leading coordinates and interpret the last one as a forcing signal.
- Keep equations minimal. The audience should leave understanding what the forcing mode means physically.

### Box 3. Why multichannel helps here

- State that single-channel HAVOK can warn about bursts, but mHAVOK lets us test whether extra observables sharpen or stabilize the warning signal.
- Tie this directly to Hindmarsh-Rose physics: `z` is the slow adaptation variable, so it is plausible that it helps mark the burst cycle.

### Box 4. Main evidence panel

- Use one forcing-versus-burst panel that clearly shows forcing activation near burst onset.
- Preferred visual: `../plots/havok_hr_burst_detection.png`
- Caption should say what the reader is supposed to notice: the forcing-like signal activates around burst onset and can be thresholded into warning events.

### Box 5. One supporting mHAVOK result

- Use only one summary figure from the new HR mHAVOK sweep:
- `../plots/mhavok_hr/mhavok_hr_x_only_model_comparison.png` if you want a simple “x-only versus tuned multichannel” story.
- `../plots/mhavok_hr/mhavok_hr_channel_combo_metrics.png` if you want the stronger observable-choice message.
- Keep `../plots/mhavok_hr/mhavok_hr_rank_delay_metrics.png` in reserve for discussion or the linked report.
- Current full 80k-sample sweep takeaway: the best overall warning model on the tested grid is `z` with `delays = 50`, `rank = 5`, recall `1.000`, precision `0.5735`, and F1 `0.7290`.
- The best full-state `x+y+z` model is weaker: `delays = 50`, `rank = 9`, recall `0.7179`, precision `0.4000`, and F1 `0.5140`.
- That is the message to emphasize: observable choice matters, and “more channels” is not automatically better.

### Box 6. Takeaways and QR

- State only the defensible claims:
- HAVOK forcing is useful for burst warning.
- HR mHAVOK results show that observable choice matters; extra channels can materially change warning quality.
- Put a QR code in the top-right corner linking to the GitHub repository and the longer report.
- If space allows, pin a one-page PDF summary next to the poster instead of crowding the poster with additional result panels.

## What To Leave Off The Poster

- Do not put every heatmap, CSV table, or diagnostic figure on the poster.
- Do not spend poster space on long lists of hyperparameters.
- Keep Lorenz as a short comparison point or move it entirely behind the QR code if the poster is centered on HR mHAVOK.

## Current HR mHAVOK Outputs To Mine

- `../plots/mhavok_hr/analysis_metadata.csv`
- `../plots/mhavok_hr/rank_delay_event_metrics.csv`
- `../plots/mhavok_hr/all_channel_combo_metrics.csv`
- `../plots/mhavok_hr/tuned_component_r2.csv`
- `../plots/mhavok_hr/mhavok_hr_x_only_model_comparison.png`
- `../plots/mhavok_hr/mhavok_hr_channel_combo_metrics.png`
- `../plots/mhavok_hr/mhavok_hr_rank_delay_metrics.png`

## Current Numbers To Cite Verbally

- Full HR mHAVOK sweep size: `80,000` post-transient samples with `39` burst onsets.
- Warning metric used in the sweep: a `20` time-unit warning window with forcing activations merged across gaps shorter than `30` time units.
- Best overall warning combo on the tested grid: `z` only.
- Best full-state combo on the tested grid: `x+y+z`, but it does not beat the simpler `z` or `x+z` warning models.