# 16B-1 — Actual Acquisition and Subset Freeze

## What is frozen here

16A froze which datasets enter Stage 16. This implementation freezes the next
layer: how an acquired archive becomes an immutable experimental input.

The repository does not contain third-party raw data. Acquisition happens
outside Git, using the committed source URL/version/checksum and the acquisition
utility. The resulting acquisition manifest is then committed.

The acquisition gate is:

    source/version
        -> download exact archive
        -> verify publisher checksum
        -> compute local SHA-256
        -> inventory archive
        -> deterministic subset selection
        -> freeze exact member paths
        -> 16B-1 raw-IQ ingestion
        -> 16B-2 temporal characterization

## Paper-1 physical set

The minimum physical set remains:

- INRIA I/Q RF Fingerprinting / PLA — DOI 10.5281/zenodo.18268648, v1.0.0.
  Source: https://zenodo.org/records/18268648
- LoRadar — DOI 10.5281/zenodo.16302856, v1.
  Source: https://zenodo.org/records/16302856

The current records describe the INRIA data as raw I/Q collected with a
BladeRF AX4/GNU Radio, and LoRadar as raw complex64/interleaved-float32 I/Q
from real satellite overpasses with varying SNR and Doppler.

The landing-page license field for these two records remains unverified in the
16A audit. Acquisition therefore records the source and checksum but does not
make a redistribution claim. Raw bytes remain outside the repository.

## Acquisition utility

experiments/16B_1_acquisition.py supports:

- explicit source URL and exact release;
- streamed download in 8 MiB chunks;
- publisher MD5 verification before extraction;
- local SHA-256 over the exact archive;
- ZIP inventory without changing source bytes;
- deterministic member selection;
- emitted JSON acquisition/subset manifest.

The utility is deliberately generic rather than hard-coding a hidden sampling
choice.

## Subset-freeze rule

The default selector ranks eligible raw signal members using:

    SHA256(f"{seed}:{member_path}")

and takes the first N ranked members.

Therefore selection is:

- deterministic;
- independent of ZIP member order;
- reproducible across machines;
- auditable from the seed and inventory;
- frozen once the emitted manifest is committed.

For the actual Paper-1 run, the selector is not allowed to be used as a
post-hoc performance optimizer. Dataset-specific stratification constraints
must be expressed before evaluation, and the final selected paths/counts are
committed before MIN/baseline results are inspected.

### INRIA

The primary rule from 16A is preserved:

- include all released device IDs when feasible;
- split by device/session/file identity;
- never allow fragments of the same recording to cross train/validation/test;
- if compute limits require a cap, apply the same deterministic balanced cap
  to every device and commit exact selected paths.

### LoRadar

The primary rule from 16A is preserved:

- cover the documented PHY configurations;
- cover multiple collection dates/sessions;
- cover available SNR/Doppler regimes;
- select whole session/packet units rather than arbitrary byte ranges;
- commit the exact file/packet selection before evaluation.

The generic ZIP ranker is therefore a freeze mechanism, not a substitute for
the dataset-specific stratification logic. The final LoRadar subset manifest
must contain the actual configuration/session strata.

## What CI proves

CI uses only a tiny generated fixture. It does not download INRIA or LoRadar,
because downloading large third-party archives in every CI run would make
reproducibility depend on external service availability and waste compute.

The CI proof is limited to:

1. deterministic archive inventory;
2. order-independent selection;
3. checksum rejection;
4. deterministic acquisition manifest generation.

Actual physical acquisition is an explicit operator step and must produce a
committed manifest containing the local SHA-256 and exact selected members.

## Raw-data policy

Never commit:

- ZIP archives;
- extracted raw .bin files;
- credentials/tokens;
- generated temporary downloads.

Commit:

- source/version/checksum metadata;
- acquisition timestamp;
- local SHA-256;
- archive inventory summary;
- selected member paths;
- subset seed/rule;
- resulting 16B-1 recording manifests.

## Gate to 16B-2

16B-2 starts only after the physical acquisition manifests are frozen. It
consumes the exact selected raw recordings and produces temporal measurements;
it does not alter the acquisition decision.
