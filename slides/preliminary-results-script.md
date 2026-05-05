# Preliminary Results Presentation Script

## Slide 1: Preliminary HAVOK Results

"Our project is looking at whether the HAVOK framework can recover meaningful transition structure in a neural system that shows chaotic bursting. Instead of working with Lorenz, we are now using the Hindmarsh-Rose model, which is a standard synthetic neuron model with rich nonlinear dynamics. The specific question we want to ask is whether the HAVOK forcing signal can help identify or predict the onset of bursting events when we only observe a single scalar signal. For this preliminary pass, we trained a HAVOK model with rank 11, 100 delays, and 100,000 samples from the Hindmarsh-Rose simulation."

## Slide 2: Setup And Framing

"Here the key setup is that we generate the full Hindmarsh-Rose trajectory, but we only feed the scalar signal x of t into HAVOK. So even though the full system is three-dimensional, the model only sees one observed coordinate and has to reconstruct structure through time-delay embedding. That makes this a good test of whether HAVOK can extract low-dimensional transition dynamics from partial observations. The goal is not just to fit the time series well, but to see whether the forcing coordinate isolates the moments when the system enters bursting behavior."

## Slide 3: Result 1: Reconstruction Is Stronger Than Forecasting

"The first result is that reconstruction is clearly stronger than long-range forecasting. On the training signal, the HAVOK reconstruction tracks the observed burst structure reasonably well, with a correlation of about 0.87 and a relative RMSE of about 0.54. When we extend to prediction using forcing from a longer run, the model still captures the general qualitative pattern of the bursting dynamics, but the accuracy drops, with correlation around 0.76. So the main takeaway from this slide is that the delayed linear model is doing a decent job representing the local geometry of the dynamics, but phase errors accumulate when we try to push it farther forward in time."

## Slide 4: Observation / Interpretation

"This slide gets closest to our research question. The HAVOK forcing signal is sparse, which is important, because it means most of the trajectory is being handled by the near-linear delayed subsystem and only certain transition periods need extra input. In this run, the forcing is active only about 9.7 percent of the time. More importantly, every detected burst onset has a thresholded forcing event within 0.6 seconds, and the median nearest-event gap is only about 0.12 seconds. That said, the timing is not yet strong enough to claim robust early prediction, because the nearest forcing event happens before onset for only about 36 percent of bursts. So the cautious interpretation is that the forcing signal is already a strong marker of burst transitions, but it is not yet consistently an early-warning predictor."

## Slide 5: Result 3: The Embedding Looks Very Low Rank

"The third result is that the Hankel embedding appears to be extremely low rank. The first five singular values capture essentially all of the energy in the Hankel matrix, which tells us that the delayed-coordinate representation is very compressible. That is encouraging, because it suggests the observed bursting dynamics really do lie on a relatively compact structure that HAVOK can exploit. At the same time, the learned operator is only approximately in the ideal skew-symmetric, banded form, so the model is not perfectly matching the cleanest theoretical picture. That likely helps explain why reconstruction is better than longer-horizon prediction."

## Slide 6: Expanded Preliminary Results

"Putting these pieces together, our current preliminary result is that HAVOK is doing something meaningful on the Hindmarsh-Rose system. From only a scalar observation, it builds a compact delayed model that reconstructs the trajectory reasonably well, isolates sparse forcing events, and keeps those forcing events tightly coupled to burst transitions. So the strongest claim we can make right now is not that we have solved burst-onset prediction, but that HAVOK appears to separate smooth bursting evolution from transition events in a way that is scientifically interpretable. The next natural step is to convert that into a real prediction benchmark by quantifying lead time, detection accuracy, and then comparing those numbers to the Lorenz case from the original HAVOK paper."

## Short Closing If Needed

"So overall, this version of the project is stronger than the earlier synthetic-neural idea because we now have a concrete dynamical system, a clear event definition, and a focused question: whether the HAVOK forcing coordinate contains predictive information about burst onset. Our preliminary answer is that it definitely tracks those transitions, and the next step is to test whether that tracking can be turned into a competitive predictor."