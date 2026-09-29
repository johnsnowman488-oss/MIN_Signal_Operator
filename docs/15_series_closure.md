# 15-Series Experimental Closure

The 15-series is now closed. The experiments are retained as individual CI-backed evidence, while this document and `notebooks/15A_15F_unified_series_closure.ipynb` provide the synthesis layer.

## Evidence chain

- 15A–15B: temporal geometry, process alignment, and task-optimal timescale interventions.
- 15C: matched-timescale IIR control.
- 15D: equal-state representation comparison.
- 15E: actual state-budget sweep, N = 1, 2, 4, 8, 16.
- 15F-A: first synthetic communications transfer.
- 15F-B: modulation/channel stress.
- 15F-C: communications state-budget scaling.
- 15F-D: final memory-geometry closure.

15F-D CI run: `36635603200`  
Artifact: `experiment-15f-d-results`  
Artifact SHA-256: `6a1ec308c7ae447fc343f6dee52a61428161887106cf68bd888b4a9a39e3ff74`  
Repository SOE equivalence relative error: `1.4390190635778494e-11`

## 15F-D protocol

Five channel families were fixed before analysis:

1. sparse symbol-spaced taps
2. dense symbol-spaced taps
3. diffuse exponential delay spread
4. diffuse gamma delay spread
5. continuous multiscale delay spread

All were evaluated across the same process, scale, SNR, modulation, seed, and state-budget grid. The representation constructors were reused directly from 15E. The communications readout retained the established 512-symbol training / 512-symbol held-out test split.

The horizon control was deliberately bounded to a reference slice: N=16 and SNR=10 dB. Its FIR tap count was determined analytically from the MIN kernel's 95% integrated temporal mass and converted to symbol-spaced taps; it was not fitted to the observed channel.

## What 15F-D established

The equal-state result does not show universal MIN superiority.

- **Sparse taps:** FIR remains strongly matched to the channel geometry.
- **Dense taps:** MIN is competitive and lower than FIR at intermediate budgets, but FIR becomes much better at N=16.
- **Diffuse exponential:** MIN is lower than FIR at N=4, while FIR becomes lower at N=8 and N=16.
- **Diffuse gamma:** the representations remain close; FIR is slightly lower at N=16.
- **Continuous multiscale:** MIN is lower than FIR at N=2 and N=4, while FIR becomes lower at N=8 and N=16.

The horizon-matched N=16 reference control gives mean BER of approximately 0.0234, 0.0838, 0.2331, 0.4938, and 0.2092 for sparse taps, dense taps, diffuse exponential, diffuse gamma, and continuous multiscale respectively. On that same reference slice, MIN-16 is below the horizon-matched FIR for the three smooth continuous families.

## Final 15-series interpretation

The defensible result is not “MIN is a better filter.”

The evidence supports:

> **Temporal memory geometry and state allocation jointly determine representation efficiency and task accessibility.**

MIN can be competitive with finite-delay representations for some distributed-memory channels, and the horizon-matched control gives a concrete conditional example. The experiments do not establish universal superiority, real-world communications superiority, or optimality of MIN state allocation.

This closes the 15-series and leaves 16 with a clean next question: whether the memory-geometry/state-accessibility relationship survives a more controlled complex-IQ setting.
