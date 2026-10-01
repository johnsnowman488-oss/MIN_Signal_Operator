# 16A — Physical-IQ Dataset Manifest

## Purpose

Stage 16 begins with a **dataset manifest, not a bulk download**.

This document freezes the candidate datasets, their provenance, representation, licensing status, metadata available for experiment design, deterministic subset rules, publisher-provided hashes, and the exact reason each dataset is assigned to Tier 0, Tier 1, or Tier 2.

No raw dataset is stored in this repository. The manifest is the source of truth for acquisition and later ingestion.

**Manifest audit date:** 2026-10-01  
**MIN repository baseline:** `1eeb08fdb14ad2fefffe87d742af4ef0d89a564f`

## Tier policy

| Tier | Role | Required for Paper 1? |
|---|---|---|
| Tier 0 | Historical/synthetic control and regression bridge. Must preserve comparability with the 15-series and provide known labels/conditions before physical data is introduced. | Yes, as a control layer |
| Tier 1 | Genuine recorded RF/IQ data used for the physical reality check. At least two materially different physical regimes are required before making a physical-data generalization. | Yes |
| Tier 2 | Specialized physical stress/validation material that can expose additional failure modes but is not required for the minimum Paper-1 dataset set. | No; reserve |

## Dataset register

### Tier 0 — RadioML 2016.10A

- **DOI:** Not assigned on the current DeepSig historical dataset page.
- **Exact release identifier:** RadioML 2016.10A (historical dataset from 2016).
- **Source:** DeepSig historical datasets page.
- **License status:** **Verified — CC BY-NC-SA 4.0**, per DeepSig's current license notice covering the datasets on that page.
- **Data size / structure:** Synthetic GNU Radio dataset; 11 modulation classes (8 digital, 3 analog); varying SNR; Python pickle.
- **Sample representation:** Dataset examples are signal arrays in the historical RadioML pickle format; treat the exact dtype/layout as an ingestion-time assertion rather than assuming a complex64 wire format.
- **Metadata/labels available:** modulation class; SNR; example/signal index as implied by the dataset organization.
- **Subset-selection rule:** Deterministic stratification over every available modulation and SNR cell used in Stage 15/16 regression checks. Record exact selected indices after acquisition; never sample ad hoc by file order.
- **Publisher-provided hashes:** None exposed on the current DeepSig dataset page; local SHA-256 must be recorded for the exact acquired archive/file at ingestion.
- **Precise reason for Tier 0:** It is synthetic and historical, so it cannot establish physical validity. Its value is controlled continuity with the existing 15-series and a cheap regression/control layer for catching implementation regressions before physical-data experiments.

### Tier 0 — RadioML 2018.01A

- **DOI:** Not assigned on the current DeepSig historical dataset page.
- **Exact release identifier:** RadioML 2018.01A (historical dataset from 2017).
- **Source:** DeepSig historical datasets page.
- **License status:** **Verified — CC BY-NC-SA 4.0**, per DeepSig's current license notice covering the datasets on that page.
- **Data size / structure:** Synthetic dataset with simulated channel effects; 24 digital/analog modulation types; 2,000,000 examples; each example is 1,024 samples; HDF5.
- **Sample representation:** Complex floating-point values in HDF5.
- **Metadata/labels available:** modulation type and dataset conditions including SNR as part of the historical benchmark organization.
- **Subset-selection rule:** Deterministic, stratified sample across modulation and SNR cells. Use a fixed manifest of example indices rather than random sampling at runtime.
- **Publisher-provided hashes:** None exposed on the current DeepSig dataset page; local SHA-256 must be recorded for the exact acquired archive/file.
- **Precise reason for Tier 0:** It extends the synthetic control layer from 11 to 24 modulation types and introduces simulated channel effects while remaining non-physical. It is a broader regression bridge, not evidence that MIN works better on real RF.

### Tier 1 — INRIA I/Q Signal Dataset for RF Fingerprinting and Physical Layer Authentication

- **DOI:** 10.5281/zenodo.18268648
- **Exact version:** v1.0.0
- **Zenodo publication:** 2026-01-16
- **License status:** **Unverified / blocking for redistribution claims.** The current Zenodo landing page renders the Rights → License field without a license value. Do not infer a permissive license. Before any repository redistribution, confirm the archive README/license and/or obtain explicit permission if needed.
- **Publisher archive:** `PLA_dataset.zip`
- **Publisher-provided archive MD5:** `aff583bee6f4efccd08fe78c731bf03d`
- **Archive size shown on record:** 710.7 MB
- **Underlying data volume shown by record:** 135.0 GB
- **Format:** Raw binary I/Q, interleaved 32-bit floats, headerless; complex64-equivalent loading in NumPy.
- **Acquisition:** BladeRF AX4 SDR + GNU Radio; COTS wireless devices including NRF52840-class devices; 20 Msps; 2.4 GHz ISM band.
- **Organization/metadata:** Raw bursts organized by Device ID; example archive layout exposes device directories and trial/burst filenames.
- **Signal structure explicitly documented:** Each file captures a device burst including transient turn-on and steady-state phases; the dataset specifically targets the transient phase.
- **Metadata/labels available for experiment design:** device identity; file/burst identity; transient vs steady-state region implied by the capture structure; acquisition hardware and fixed RF settings from the dataset description.
- **What is *not* assumed:** No modulation/BER/EVM ground truth is asserted by the manifest unless the acquired archive metadata explicitly provides it.
- **Subset-selection rule:** Use the complete set of device IDs available in the released archive for the primary physical experiment; split by device/session/file identity so no burst-level fragments cross train/validation/test boundaries. For compute-limited runs, use a deterministic balanced cap per device and preserve the cap list in a derived acquisition manifest.
- **Local hash requirement:** After acquisition compute SHA-256 for the downloaded archive and, where extracted data are used, hashes for each source file or a Merkle/index manifest.
- **Precise reason for Tier 1:** This is genuinely captured SDR/IQ from a terrestrial 2.4 GHz physical radio environment with device-dependent hardware impairments and transient/steady-state dynamics. It supplies the first core physical dataset for testing whether memory geometry has measurable value outside synthetic simulations.

### Tier 1 — LoRadar: A Raw IQ Dataset of Satellite-Ground LoRa Communication Signals

- **DOI:** 10.5281/zenodo.16302856
- **Exact version:** v1
- **Zenodo publication:** 2025-07-22
- **License status:** **Unverified / blocking for redistribution claims.** The current Zenodo landing page renders the Rights → License field without a license value. Do not infer a permissive license. Confirm the archive README/license or obtain explicit permission before redistribution.
- **Publisher files:** 18 split ZIP parts plus README.md.
- **Publisher-provided README MD5:** `9fef031bd6cdec3ea1aa38e9dfc9238a`
- **Publisher-provided ZIP-part MD5s:** Recorded in `16A_DATASET_MANIFEST.json` for parts 001–018.
- **Data organization:** 181 raw session files; ~127 GB raw data described in README; 6,664 valid LoRa packets; collected 2025-05-05 through 2025-05-17.
- **Format:** Headerless raw `.bin`; Complex64 (float32 I and float32 Q), interleaved [I0,Q0,I1,Q1,...].
- **RF/acquisition:** 4 MHz sampling; 399–403 MHz; single LoRa ground terminal; HackRF SDR; real LEO satellite overpasses.
- **Physical-layer configurations documented in README:** center frequency, bandwidth, spreading factor; five detected configurations including 125/62.5/250 kHz bandwidths and SF10/SF11.
- **Physical conditions:** varying SNR and Doppler; packet sparsity ~3% of session time.
- **Metadata/labels available for experiment design:** session/file identity; collection date; physical-layer configuration; detected packet identity/locations where recoverable from README/data; SNR/Doppler as available from released metadata or derived signal analysis.
- **Subset-selection rule:** Do not download all parts by default. Construct a deterministic physical-validation subset stratified by (a) all five documented PHY configurations, (b) multiple collection sessions/dates, and (c) SNR/Doppler regimes. Prefer whole-session or whole-packet units so temporal correlations are not broken. The exact part/file selection and sample counts become a committed derived manifest before the run.
- **Local hash requirement:** Verify every acquired ZIP against the publisher MD5; then compute SHA-256 for the selected archives and extracted source files used by experiments.
- **Precise reason for Tier 1:** It is an independent physical regime from INRIA: satellite-ground communications, different carrier band, HackRF hardware, Doppler, long-duration sessions, and real LoRa traffic. It tests whether any observed MIN behavior survives a major change in propagation and acquisition conditions rather than being a property of one terrestrial dataset.

### Tier 2 — CYGNSS Level 1 Raw Intermediate Frequency Data Record

- **Identifier:** 10.5067/CYGNS-L1RIF
- **Version status:** The current NASA Open Data record does not expose a simple semantic version in its title. Treat the dataset identifier plus acquisition/catalog timestamp as the exact source reference until a specific archive/version is selected.
- **License status:** **Catalog metadata points to US Government Works** via `https://www.usa.gov/government-works`. This is recorded as source metadata, not as a blanket legal conclusion for every ancillary file or derived artifact.
- **Format:** Multiple public resources including BIN plus associated documentation/metadata formats. The core signal product is raw IF sensor data, not a ready-made complex64 IQ stream.
- **Signal structure:** Raw sensor counts from the CYGNSS Delay Doppler Mapping Instrument; three input antenna channels; records are 30–90 s, typically 60 s; commanded to coincide with spacecraft overpasses.
- **Metadata:** spacecraft/overpass context, target area/time, antenna channel, raw IF record identity, and associated catalog metadata.
- **Hash note:** The current NASA catalog exposes a `source_hash` for its metadata record (`f07a0b4a93d96ecd24d5f0b3d474f016b0ff5a3039868b9f3c5a047601893581`), not a verified hash of every binary data resource. Data-file SHA-256 must therefore be generated locally after acquisition.
- **Subset-selection rule:** If activated, select a small number of complete raw-IF records spanning different spacecraft passes and antenna channels; preserve full records rather than arbitrary windows for the first stress test.
- **Precise reason for Tier 2:** It provides very raw, instrument-level dynamic RF data that can stress an ingestion pipeline and memory representation outside conventional terrestrial communications. It is valuable as a specialized robustness test but is not the cleanest Paper-1 communications comparison because the signal representation is instrument-specific raw IF counts rather than directly comparable complex baseband IQ.

## Acquisition and hashing policy

For every acquired source:

1. Record the exact source identifier, version, URL/DOI, acquisition timestamp, and source-reported checksum.
2. Verify all publisher-provided checksums before extraction.
3. Compute and store a local SHA-256 for the exact archive used.
4. Do not modify raw source bytes. Conditioning, slicing, normalization, synchronization, and conversion are derived stages with their own versioned manifests.
5. Raw data are not committed to Git. Only manifests, code, small license-compatible fixtures, and derived summary artifacts belong in the repository.
6. Any redistribution statement in the paper must use the verified license state at the time of publication, not this initial audit alone.

## Paper-1 minimum set

The minimum Stage-16 evidence set is:

**Tier 0:** RadioML 2016.10A + RadioML 2018.01A  
**Tier 1:** INRIA I/Q Signal Dataset + LoRadar  
**Tier 2:** CYGNSS Raw IF remains optional

This set is deliberately small. The design objective is coverage of distinct signal regimes, not the largest possible dataset count.

## Gate before 16A ingestion implementation

Physical acquisition code must not be treated as complete until the selected archive/version is frozen, license status is recorded, publisher checksums are verified, and the exact derived subset manifest is committed.
