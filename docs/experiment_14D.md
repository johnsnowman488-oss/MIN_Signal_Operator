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

    λ = 0, 10^-10, 10^-8, 10^-6, 10^-4, 10^-2, 10^-1.

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
- 7 ridge strengths;

giving 37,800 full-state readout rows, plus 900 scalar-MIN rows.

The held-out target symbols are never used to fit the kernel, PCA basis, or
readout.

## Executed result

The authoritative final pull-request execution completed successfully:

- Experiment 14D workflow: 36221414451 (run #12, completed successfully)
- final experiment head: 3932e88bc2eb9ed97cc67a6fd7f6a1506994d3e3
- result artifact: experiment-14d-results (artifact ID 10899102617)
- artifact SHA-256: 50a7569764843e2b8848b038077843010d544d03730f17a7d5e50dd671de6950

The authoritative run generated 300 underlying task cases, 37,800 full-state readout
rows, and 900 scalar-MIN rows. Repository-state equivalence remained
1.61e-11 relative error. The artifact was produced from the same commit that
contains the final 14D least-squares, result-schema, and regularization fixes.

At full dimension d=16, mean held-out NMSE across all task cases was:

| Geometry | λ=0 | Best tested λ |
|---|---:|---:|
| clustered | 0.6918 | 0.6918 at λ=0 |
| logspread | 0.7252 | 0.5321 at λ=10^-10 |
| wide | 1.9091 | 0.3230 at λ=10^-6 |

The best regularization scale is geometry- and condition-dependent rather than
uniform. For d=16, the best tested λ by SNR was:

- clustered: λ=0 at 0, 10, 20 and 30 dB;
- logspread: λ=10^-10 at 0, 10 and 20 dB, λ=0 at 30 dB;
- wide: λ=10^-6 at 0 and 10 dB, λ=10^-10 at 20 and 30 dB.

The augmented full-state readout condition number differed sharply by geometry:
approximately 1.42e15 for clustered, 8.82e9 for logspread, and 6.53e6 for
wide. Under the κ>10^12 diagnostic threshold, all clustered cases were flagged
while the logspread and wide cases were not.

The scalar-MIN secondary path remained comparatively insensitive to geometry:
mean NMSE was approximately 0.9669, 0.9794 and 0.9990 for clustered,
logspread and wide respectively.

These results support the intended 14D interpretation: downstream numerical
accessibility is a substantial part of the observed geometry/task interaction,
especially for the highly redundant clustered state. At the same time,
regularization does not collapse all geometry differences into a common curve;
logspread and wide retain different dimension/regularization behavior.

The λ values above are **relative ridge strengths** normalized by the training
Gram scale. Therefore the λ=10^-10 entry is a near-zero normalized regularizer,
not a claim of numerical identity with 14C's absolute 10^-10 Gram penalty.

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
