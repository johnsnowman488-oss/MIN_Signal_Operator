# Stage 16B-1 — Raw Physical-IQ Ingestion and Provenance

## Purpose

Stage 16A froze the minimum physical/synthetic dataset set and acquisition
policy. **16B-1 is the first executable bridge from that manifest to actual
recording bytes.**

The question here is not whether MIN works. The question is:

> Can a selected recording be loaded deterministically, checked for basic byte/
> sample integrity, and tied to immutable provenance before any conditioning or
> temporal analysis occurs?

This is an ingestion gate. It deliberately stops before 16B temporal
characterization and before any MIN-vs-baseline comparison.

## Input contract

The first physical Tier-1 datasets selected for Stage 16 contain headerless
binary IQ represented as interleaved float32 I/Q samples. The loader therefore
uses:

    [I0, Q0, I1, Q1, ...] -> complex64

The implementation is little-endian (<f4) for the float32 wire values.

The code does not silently repair malformed streams. An IQ file whose byte
length is not a multiple of 8 bytes per complex sample is rejected.

## Large-file handling

Two different operations are kept explicit:

1. The sample loader seeks directly to a requested sample interval, so a small
   slice does not require loading the full recording.
2. The file-integrity scanner reads the raw file in fixed-size chunks, so the
   full-record integrity pass is bounded in memory even for multi-GB archives.

The source SHA-256 is also computed incrementally over the complete raw bytes.

This matters for the Stage-16 candidates because the released physical archives
range from hundreds of MB to multi-GB/TB-scale collections; 16B-1 must not make
whole-file materialization part of the ingestion contract.

## Provenance contract

Every selected recording must carry the following fields:

- recording identity and dataset identity;
- dataset tier;
- DOI/source identifier and exact source version;
- sampling rate and center frequency;
- wire/sample representation;
- publisher/source checksum value as recorded in 16A;
- conditioning version;
- derived subset identity;
- split identity.

The emitted recording manifest additionally records:

- SHA-256 of the exact source bytes;
- byte size and complete sample count;
- number of samples actually scanned/read;
- duration implied by the declared sample rate;
- wire dtype/layout;
- finite/non-finite count;
- I/Q means and RMS values;
- Q/I power ratio where defined;
- mean power and peak magnitude;
- chunk size used by the integrity scanner;
- whether the read was intentionally truncated for a bounded smoke test.

## Integrity checks

16B-1 reports, but does not correct:

- non-finite samples;
- zero-length inputs;
- DC mean in I and Q;
- I/Q RMS and power imbalance;
- peak magnitude.

These values are diagnostics only. No normalization, DC removal, imbalance correction,
clipping repair, synchronization, filtering, or gain adjustment occurs here.

## Hashing rule

The SHA-256 is computed over the exact source file bytes. For a complete raw
recording, the hash covers the entire file. For a bounded smoke test, the
source hash still covers the complete source file, while the manifest marks
that only a sample prefix was scanned.

Publisher-provided checksums from 16A remain provenance fields; 16B-1 does not
claim that a local SHA-256 is equivalent to a publisher MD5 or other source
checksum.

## Deterministic CI path

The CI workflow uses an internally generated, deterministic fixture rather than
downloading any external dataset. This proves the ingestion path without
putting third-party raw data into Git or CI.

Run locally:

    python experiments/16B_1_iq_ingestion.py --self-test

Run the targeted tests:

    pytest -q tests/test_experiment_16b1.py

## Boundary to 16B-2

16B-2 should consume the immutable recording manifests produced here and begin
the physical temporal characterization:

    ACF -> PSD -> coherence/memory timescales -> nonstationarity -> impairment/regime map

No MIN geometry selection belongs in 16B-1.
