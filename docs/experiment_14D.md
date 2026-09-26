# Experiment 14D — Representation Accessibility and Decoder Conditioning

## Aim

14D asks whether the geometry-dependent task effects observed in 14C are
properties of the MIN/SOE representation itself or are partly produced by the
downstream decoder and its numerical conditioning.

The controlled chain is:

    kernel geometry → MIN/SOE state → PCA/readout → held-out symbol recovery

The observation, environment, signal, SNR, and geometry are held fixed while
the readout is varied.

## Protocol

The same 300 underlying cases as 14C are used:

- BPSK, QPSK, 16-QAM;
- five controlled colored-noise environments;
- SNR = 0, 10, 20, 30 dB;
- five seeds;
- clustered, logspread, and wide 16-mode geometries.

For each geometry, the full 16-mode state is fitted on training-symbol centers
and evaluated on held-out centers after PCA projection to:

    d = 1, 2, 4, 8, 12, 16.

Each retained dimension is decoded with ridge strengths:

    λ = 0, 10^-8, 10^-6, 10^-4, 10^-2, 10^-1.

The ridge penalty is normalized by the mean diagonal scale of the augmented
training Gram matrix, while the intercept is not regularized. This makes the
regularization parameter a relative rather than raw-data-scale quantity.

A secondary scalar-MIN path is retained at λ=10^-6.

## Conditioning measurements

For each geometry/case the augmented full-state readout design records:

- largest singular value;
- smallest positive singular value;
- numerical rank;
- condition number;
- an ill-conditioned flag at κ > 10^12.

These are diagnostics of decoder accessibility, not measurements of information
content by themselves.

## Why 14D follows 14C

14C showed that matched L=16 and D_eff=256 geometries can have substantially
different full-state task behavior and D_task. 14D therefore tests whether
those differences survive controlled changes in the downstream readout.

Three broad outcomes are informative:

1. **Regularization closes geometry gaps.** Some 14C differences were decoder
   conditioning effects.
2. **Regularization leaves geometry gaps.** This strengthens the interpretation
   that the representation itself carries geometry-dependent task structure.
3. **Dimension curves converge while absolute NMSE remains different.** Geometry
   primarily changes compression/accessibility rather than only raw task
   accuracy.

No geometry is ranked. The purpose is causal separation of representation
structure from decoder sensitivity.

## Data accounting

There are:

- 300 underlying task cases;
- 3 geometries;
- 6 retained dimensions;
- 6 ridge strengths;

giving 32,400 full-state readout rows, plus 900 scalar-MIN rows.

The held-out target symbols are never used to fit the kernel, PCA basis, or
readout.

## Relation to the program

13A: controlled kernel geometries.

13B: geometry → state structure.

13C: environment → kernel construction and mismatch.

14A: task-relevant information in the environment-informed full state.

14B: kernel alignment and scalar MIN exposure.

14C: geometry → task-relevant state structure.

14D: decoder accessibility, conditioning, and stability.

14E can then move to a second task such as prediction, environment
identification, or change detection without carrying an unresolved
decoder-conditioning confound forward.
