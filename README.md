# MIN Signal Operator

A research laboratory for the **Memory-Integrated Network (MIN) signal operator**: a structured causal temporal-memory representation built from finite sums of exponential memory modes.

The project investigates a deliberately open question:

> **When does structured temporal memory provide a useful representation of an information-bearing signal, and how does the geometry of that memory affect what can be recovered or learned from the resulting state?**

The repository is experimental research software. It does **not** assume that MIN is superior to conventional filters, finite history, state-space models, or learned temporal representations. Advantages must emerge from controlled comparisons.

## Core operator

The continuous causal MIN operator is

$$
(Mf)(t)=\int_0^t K(t-s)f(s)\,ds.
$$

A finite sum-of-exponentials (SOE) kernel,

$$
K(\tau)=\sum_{j=1}^{L} w_j e^{-\gamma_j\tau},
$$

gives a finite recursive memory state. In discrete time, the representation can be viewed schematically as

$$
x_{0:t}\longrightarrow q_t,
$$

where the state geometry depends not only on the nominal number of modes $L$, but also on decay-rate placement, weights, signal excitation, observation conditions, and the task.

The current research therefore treats MIN primarily as a **structured temporal representation operator**, rather than simply as another filter.

## Research progression

The experiments have evolved through several distinct questions:

1. **Operator verification** — verify the causal memory operator numerically.
2. **SOE realization** — connect convolutional memory to finite recursive states.
3. **Kernel identification** — compare Prony, Matrix Pencil, ESPRIT, Vector Fitting, and positive NNLS representations.
4. **Signal atlas** — measure how different kernels transform elementary and digital signals.
5. **Frequency and finite-horizon analysis** — separate genuine memory effects from discretization, startup, and truncation effects.
6. **Communication transformation and recovery** — test BPSK/QPSK/16-QAM under controlled memory, AWGN, multipath, and receiver models.
7. **Receiver recoverability** — compare FIR, IIR/state-space, SOE/MIN-aware, and explicit SOE-state inversion paths.
8. **Conditioning and regularization** — distinguish forward representation from inverse stability.
9. **Kernel/SOE geometry atlas** — vary mode count, decay-rate geometry, and weights under controlled excitation.
10. **State geometry** — measure how kernel geometry becomes realized temporal state geometry.
11. **Environment-conditioned kernels** — map controlled temporal environments to SOE kernels and test finite-observation estimation and mismatch.
12. **Task-level evaluation** — ask whether representation geometry changes task accessibility.
13. **Predictive representation** — test whether MIN states expose temporal information unavailable to instantaneous observations.
14. **Next stage: task-matched memory design** — compare fixed, task-matched, and learned memory geometries against finite-history and standard state-space baselines under approximately matched computational budgets.

The project intentionally separates:

$$
\text{environment}
\rightarrow
K_{env}(t)
\rightarrow
\{\gamma_j,w_j\}
\rightarrow
\text{MIN state geometry}
\rightarrow
\text{task accessibility}
\rightarrow
\text{task performance}.
$$

The purpose is to determine whether this chain can become a reproducible design principle rather than a post-hoc interpretation.

## What has been established so far

### 1. MIN/SOE numerical foundation

Experiments 01–05 establish the numerical operator, finite-SOE realization, non-exponential stress testing, positive NNLS approximation, and comparisons among standard SOE identification methods.

A key methodological point is that identification failure can result from model mismatch or poor conditioning without invalidating the underlying memory operator.

### 2. Memory changes signals nontrivially

Experiments 06–11 establish that MIN memory produces measurable temporal transformation, frequency-dependent behavior, finite-horizon effects, and substantial intersymbol mixing in the tested digital-signal settings.

The communication experiments do **not** establish a communication advantage for MIN. They provide controlled conditions for studying transformation and recoverability.

### 3. Forward representation is different from inversion

Experiments 12A–12D show that:

- generic receiver complexity does not automatically remove long-memory residuals;
- explicitly supplying SOE/MIN structure does not automatically improve recovery;
- a matched noiseless SOE state can represent/invert the tested system to numerical precision;
- direct inversion can nevertheless become severely ill-conditioned under noise and channel-estimation error;
- regularization changes the practical inverse problem.

This motivates a separation between:

$$
\text{forward representation}
\neq
\text{inverse stability}
\neq
\text{task utility}.
$$

### 4. Memory geometry is an independent representation variable

Experiments 13A–13C establish a more general representation branch.

The central hierarchy is

$$
L \neq D_{eff} \neq D_{basis} \neq D_{state},
$$

and later experiments add task-accessible dimension as another distinct quantity.

Experiment 13B found a strong relationship between finite-horizon SOE basis participation and signal-driven state participation in the tested setting (Pearson $r\approx0.966$), while $D_{eff}$ was not interchangeable with realized state dimension.

Experiment 13C extends this to an environment-conditioned construction:

$$
\text{environment}
\rightarrow
K_{env}
\rightarrow
\text{SOE geometry}
\rightarrow
\text{MIN state}.
$$

Finite-observation uncertainty, rate-support mismatch, and model-order mismatch are explicitly tested rather than hidden behind an oracle kernel.

### 5. Same model order does not imply the same task representation

Experiment 14C is one of the strongest current results.

With mode count, weights, effective dimension, observed waveform, task, decoder, and evaluation protocol held fixed, changing only SOE decay-rate geometry produced materially different state/task representations.

Representative results:

| Geometry | Full-state NMSE | Task dimension |
|---|---:|---:|
| Clustered | 0.8574 | 2.04 |
| Logspread | 0.5003 | 10.20 |
| Wide | 0.7156 | 11.70 |

These numbers are specific to the controlled experiment. They are **not** a universal ranking of memory geometries.

The important result is the independence of memory geometry from nominal mode count and effective count: **where the memory time scales are placed can change the information made accessible to a downstream task.**

Experiment 14D shows that decoder conditioning is strongly geometry-dependent as well, so representation and numerical conditioning must be analyzed together.

### 6. Negative results are part of the program

Experiment 14A showed that an environment-informed MIN representation did not automatically outperform the raw noisy observation baseline. Experiment 14B showed that environment/kernel alignment does not automatically produce task-optimal representations. Experiment 14E found no general predictive advantage in the tested IID digital-symbol setting.

These results prevent the project from turning a representation hypothesis into an assumed performance claim.

### 7. MIN can expose hidden predictive information

Experiment 14F uses a genuine hidden-memory process in which the scalar observation is temporally dependent while the underlying hidden state evolves Markovianly.

Representative held-out prediction results were:

| Representation | NMSE |
|---|---:|
| Current observation | 0.1639 |
| Best MIN representation | 0.1101 |
| Best finite history | 0.0543 |
| Oracle hidden state | 0.0182 |

The result supports a narrower claim: **a causal MIN state can expose predictive information that is not available from the instantaneous observation alone.**

It does **not** show that MIN is better than finite history in general; in this controlled process, finite history performed better.

## Current scientific thesis

The strongest current interpretation is:

> **Temporal memory geometry is an independently meaningful representation variable. It is not determined by nominal SOE mode count or effective dimension alone, and changes in memory geometry can propagate into realized state geometry and task accessibility.**

A second supporting observation is:

> **Causal memory states can recover predictive information hidden by instantaneous observations, although conventional finite history can remain more effective in the tested regimes.**

Together these suggest a research direction in which memory kernels are designed as **representation mechanisms**, not merely as signal transformations.

## Current research question

The next decisive question is:

> **Can useful memory geometry be predicted or designed from the temporal structure of a task, and can it provide a better representation-per-compute trade-off than conventional finite history or standard state-space baselines?**

The next benchmark should therefore compare, under approximately matched budgets:

- fixed arbitrary SOE geometry;
- environment-matched geometry;
- task-matched geometry;
- learned memory geometry;
- finite-history/FIR representations;
- standard state-space or recurrent baselines.

Relevant measurements include:

- task/prediction error;
- retained state dimension;
- effective memory horizon;
- parameter count;
- MACs per sample;
- latency;
- training cost;
- robustness to kernel mismatch;
- robustness to observation noise and sampling changes.

Only if these experiments produce reproducible advantages should the project claim an engineering contribution to signal processing.

## Baseline policy

The primary MIN condition is linear, so linear baselines remain central:

$$
\text{raw/current observation}
\rightarrow
\text{finite history/FIR}
\rightarrow
\text{IIR/state-space}
\rightarrow
\text{SOE/MIN}.
$$

Learned temporal models may be added where the task requires them.

Volterra and memory-polynomial baselines are reserved for controlled nonlinear-memory conditions rather than being introduced simply to enlarge the benchmark.

All comparisons should preserve, where practical, matched signal realizations, train/test splits, computational budgets, and evaluation protocols.

## Repository structure

- **min/** — core implementation, kernels, synthetic signals, and metrics
- **tests/** — numerical correctness tests
- **notebooks/** — executable experiment and synthesis notebooks
- **experiments/** — reproducible benchmark runners and result artifacts
- **docs/** — research protocols and experiment documentation
- **cpp/** — future performance-oriented implementations

The notebooks are **not** the numerical source of truth. Reusable logic belongs in **min/**, experiment scripts generate the result tables, and notebooks synthesize or visualize those results.

## Development

The reference implementation uses NumPy.

Install the project and research dependencies with:

    pip install -e ".[research]"

Run the test suite with:

    pytest

Experiments should remain reproducible and should record the assumptions, baselines, seeds, and evaluation protocol needed to interpret their results.

## Research status

**Active research — representation/task branch.**

Experiments 01–12D establish the numerical, SOE, signal-transformation, receiver-recoverability, and conditioning foundations.

Experiments 13A–13C establish the kernel-geometry, state-geometry, and environment-conditioned representation branch.

Experiments 14A–14F move into controlled task-level evidence, including raw-observation controls, environment/kernel alignment, memory geometry, regularization, predictive testing, and hidden-memory prediction.

The current evidence supports a **structured temporal representation** interpretation of MIN. It does not yet establish universal superiority over FIR/IIR/state-space methods, finite history, learned temporal models, or other signal-processing technologies.

The project therefore remains deliberately empirical: **claims about usefulness must be earned by controlled experiments.**

## License

The MIN Signal Operator software and associated source code are released under the **GNU General Public License v3.0 only (GPL-3.0-only)**. See [LICENSE](LICENSE) for the complete license text.

Third-party dependencies, datasets, documents, and externally sourced materials remain subject to their respective licenses or terms.
