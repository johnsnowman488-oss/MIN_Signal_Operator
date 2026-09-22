# Experiment 12B — SOE/MIN-aware receiver

## Question

Does explicit knowledge of the sampled SOE/MIN memory structure reduce receiver complexity or recovery error relative to generic FIR/IIR receivers?

This experiment is deliberately narrower than Experiment 12A. It uses the two memories that have an explicit finite SOE representation:

- exponential: (K(t)=e^{-10t})
- two-scale SOE: (K(t)=0.7e^{-2t}+0.3e^{-30t})

The power-law memory is excluded from the primary 12B comparison because it is not an exact finite SOE model. It remains an important mismatch/control case for later robustness work.

## Structure-aware receiver

The receiver operates on symbol-rate observations. The known SOE parameters determine the sampled memory poles

[
r_j=e^{-gamma_jT_s}.
]

The actual rectangular-pulse sampled channel realization is then used to derive the fixed recursive denominator. Only the feedforward numerator is learned from the 128-symbol training prefix.

Two variants are recorded:

- **soe_exact** — uses the directly derived causal inverse denominator.
- **soe_stable** — reflects any inverse poles outside the unit circle before fitting, providing a stability-constrained structural control.

This is not claimed to be an optimal inverse, and it does not establish continuous-time MIN invertibility.

## Controls

The receiver ladder is:

[
mathrm{raw}ightarrowmathrm{FIR!-!7}ightarrow
mathrm{IIR!-!2}ightarrow
mathrm{SOE!-!exact}ightarrow
mathrm{SOE!-!stable}.
]

Conditions:

- BPSK, QPSK, 16-QAM
- exponential and two-scale SOE memory
- identity, flat Rayleigh, and normalized 3-tap multipath
- 0, 10, 20, 30 dB SNR
- five seeds
- 128-symbol training prefix
- 384 held-out symbols

The complete run contains (3	imes2	imes3	imes4	imes5	imes5=1800) receiver-condition rows.

## Sandbox validation

Before the final CI run, the corrected implementation was independently reproduced in the sandbox using the same signal generation, sampled memory, channel, noise, and receiver definitions.

At 30 dB, averaged over the three signals, three channels, and five seeds:

| Memory | Receiver | Mean EVM | Mean BER | Inverse-pole radius |
|---|---|---:|---:|---:|
| exponential | FIR-7 | 28.79% | 0.0693 | — |
| exponential | IIR-2 | 29.08% | 0.0667 | 0.740 |
| exponential | SOE-exact | 32.82% | 0.0977 | 0.740 |
| exponential | SOE-stable | 32.82% | 0.0977 | 0.740 |
| two-scale SOE | FIR-7 | 48.61% | 0.2161 | — |
| two-scale SOE | IIR-2 | 48.51% | 0.2089 | 0.806 |
| two-scale SOE | SOE-exact | 51.66% | 0.2405 | 0.806 |
| two-scale SOE | SOE-stable | 51.66% | 0.2405 | 0.806 |

These sandbox results do **not** yet constitute the final repository benchmark; the CI run is the reproducibility check.

## Interpretation

The first structure-aware result is a useful negative result.

Knowing the SOE structure did not automatically improve recovery. In the tested sampled realization, the fixed-structure receiver was slightly worse than the generic low-order FIR/IIR baselines at 30 dB.

For these two kernels:

[
oxed{	ext{SOE knowledge alone did not reduce recovery error in this receiver design.}}
]

This does not make the SOE representation useless. It shows that simply constraining the inverse denominator is not sufficient. Possible reasons include:

1. the sampled rectangular-pulse realization differs from the ideal continuous SOE transfer;
2. the inverse numerator is poorly conditioned under the available training sequence;
3. multipath and memory are being learned jointly;
4. the selected receiver parameterization does not exploit the SOE state realization optimally.

The equality of SOE-exact and SOE-stable results is also informative for these tested cases: the derived inverse recursion was already stable, so pole reflection did not alter the result.

## Scientific boundary

The experiment establishes behavior of a particular sampled communication realization:

[
	ext{continuous SOE kernel}
ightarrow
	ext{sampled rectangular-pulse memory}
ightarrow
	ext{symbol-rate observation}
ightarrow
	ext{receiver}.
]

It does not establish a theorem about the continuous MIN operator.

## Next test

Before moving to Experiment 13, the next useful step is a more direct **state-space matched receiver** in which the SOE memory states are explicitly estimated and the inverse problem is formulated as a state/input estimation problem. That will distinguish:

[
	ext{SOE parameter knowledge}
]

from

[
	ext{SOE state realization}.
]

Only after that comparison should the project decide whether MIN-aware structure provides a genuine computational advantage.
