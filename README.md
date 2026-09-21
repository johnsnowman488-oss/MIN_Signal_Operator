# MIN Signal Operator

A research laboratory for the Memory-Integrated Network (MIN) signal operator.

## Scope

This repository begins with a transparent reference implementation of the causal memory operator

$$
(Mf)(t)=\\int_0^t K(t-s)f(s)\\,ds.
$$

The initial implementation is deliberately small. It is intended for verification and experiments, not production performance.

## Research progression

The program is now organized around a central question:

> **What does the MIN memory transformation do to an information-bearing signal, how recoverable is that transformation, and when—if ever—does deliberately introducing such memory become useful?**

The roadmap therefore separates **forward transformation**, **propagation**, **receiver recoverability**, and **computational complexity**. It does not assume that MIN preserves signals or improves communication.

1. **Core operator verification** — verify the causal memory operator and basic numerical behavior.
2. **Kernel and spectral foundation** — characterize exponential, multi-scale, and algebraic-tail kernels and their frequency response.
3. **SOE/state realization** — represent memory kernels with finite recursive states and validate the continuous-to-discrete relationship.
4. **Kernel identification** — compare Prony, Matrix Pencil, ESPRIT, Vector Fitting, and NNLS while tracking positivity/stability structure.
5. **Synthetic signal atlas** — measure how MIN transforms elementary waveforms.
6. **Frequency and finite-horizon analysis** — distinguish startup/finite-history effects from genuine long-memory behavior.
7. **Digital communication transformation** — establish BPSK/QPSK/16-QAM distortion under controlled MIN memory.
8. **AWGN receiver compensation** — test whether conventional receiver memory can undo the transformation on held-out symbols.
9. **Multipath/fading separation** — separate ordinary propagation distortion from MIN-induced memory.
10. **Receiver recoverability ladder** — compare raw, finite FIR, longer FIR, IIR/state-space, SOE, and MIN-aware receivers using identical channel/data realizations.
11. **Matched-complexity study** — compare parameter count, state dimension, latency/memory horizon, MACs/sample, training cost, BER, EVM, and held-out residual rather than declaring a universal winner.
12. **Forward preservation study** — explicitly measure waveform preservation through propagation, rather than inferring preservation from receiver BER.
13. **Transform-domain stability study** — test whether a deliberately MIN-transformed representation is more stable under specified propagation disturbances.
14. **Nonlinear-memory extension** — only after a controlled nonlinear channel is introduced, add memory-polynomial and Volterra baselines and compare them with an explicitly nonlinear MIN extension.
15. **Robustness and identifiability** — vary memory horizon, kernel family, SNR, fading/multipath realization, training length, and model mismatch; determine when the transformation is identifiable and invertible in practice.
16. **Real IQ validation** — move to recorded IQ only after the synthetic mechanism and complexity trade-offs are reproducible.
17. **SDR experiment** — test a constrained real-time implementation after the signal/receiver hypotheses survive offline validation.
18. **Performance engineering** — optimize Python/C++/streaming implementations only after experiments identify a justified computational bottleneck.

### Interpretation ladder

The research should distinguish four increasingly strong conclusions:

[
oxed{
	ext{MIN transforms signals}
;
otRightarrow;
	ext{MIN preserves signals}
;
otRightarrow;
	ext{MIN is efficiently invertible}
;
otRightarrow;
	ext{MIN improves propagation}
}
]

A positive result at each stage must therefore be demonstrated independently.

### Current stage

Experiments 01–11 establish the numerical/operator foundation, kernel identification, signal atlas, spectral/finite-horizon behavior, digital-signal distortion, AWGN compensation, and controlled multipath/fading receiver behavior.

The current evidence supports the narrower statement that **the tested MIN kernels impose temporal memory structures that are not fully captured by short symbol-spaced FIR equalizers**. Experiment 11 does not establish irreversibility, signal preservation, or a communication advantage.

The immediate research target is therefore:

[
oxed{
	ext{MIN transform}
ightarrow
	ext{propagation}
ightarrow
	ext{receiver}
ightarrow
	ext{recoverability}
ightarrow
	ext{complexity}
}
]

with matched controls at every stage.

### Baseline policy

The present MIN operator is linear. Therefore the primary receiver comparison is:

[
	ext{FIR}
ightarrow
	ext{IIR/state-space}
ightarrow
	ext{SOE/MIN-aware receiver}.
]

Volterra and memory-polynomial models are deferred until a nonlinear-memory condition exists. Introducing them earlier would compare a nonlinear model class against a linear MIN condition without a matched task.

The project deliberately keeps **continuous MIN**, its **sampled/discrete realization**, and the **communication channel** as separate layers.

## Repository structure

- min/ — core implementation, kernels, synthetic signals, and metrics
- tests/ — numerical correctness tests
- notebooks/ — executable research experiments
- experiments/ — reproducible benchmark runners and recorded result artifacts
- docs/ — theory-to-experiment research protocol
- cpp/ — future performance implementations

## Development

The reference implementation uses NumPy. Install the project with your preferred Python environment, then run the test suite with:

```bash
pytest
```

The frequency-response experiment uses SciPy's weighted oscillatory quadrature for the long-memory power-law reference. Install the research dependencies with:

```bash
pip install -e ".[research]"
```

The notebooks are experiments rather than the source of truth; reusable logic belongs in min/ and tests should accompany numerical claims.

## Status

Early research laboratory. Experiments 01–05 establish the numerical MIN/SOE and identification foundation. Experiment 06 begins the synthetic signal atlas. Experiment 07 characterizes frequency response and explicitly separates operator discretization from finite-history effects. Experiment 08 measures the horizon required for short-memory and algebraic-tail kernels to approach their infinite-horizon response. Experiment 09 establishes that the tested memory transformations produce substantial temporal distortion in sampled digital signals. Experiment 10 tests whether that distortion is compensable by a receiver using controlled AWGN and held-out FIR equalization. Experiment 11 adds normalized flat Rayleigh and 3-tap multipath channels. Preliminary numerical reproduction shows conventional channel distortion is strongly reduced by FIR equalization, while the tested long-memory cases retain substantial residual distortion at high SNR. These observations do not establish a communication advantage for MIN.