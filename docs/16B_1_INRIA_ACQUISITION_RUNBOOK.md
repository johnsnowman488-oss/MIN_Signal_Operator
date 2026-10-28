# INRIA PLA acquisition runbook (Stage 16B-1)

## Purpose and boundary

This workflow makes the source archive retrievable and verifiable on another machine. It does not decide the final experiment subset, prove the dataset licence, or demonstrate MIN performance. Do not begin 16B-2 until the acquisition evidence and scientific suitability review are complete.

Dataset record: https://zenodo.org/records/18268648  
Version: v1.0.0  
DOI: 10.5281/zenodo.18268648  
Archive: PLA_dataset.zip  
Publisher MD5 recorded in the lock: aff583bee6f4efccd08fe78c731bf03d

The record describes hardware-captured raw I/Q data. The visible record metadata does not make the licence sufficiently clear for us to infer usage rights; check the record's machine-readable metadata and any included terms before using or redistributing the data.

## Requirements

- Python 3.10 or newer (standard library only; no pip install required).
- Network access to Zenodo over HTTPS.
- Enough disk space for the archive plus working room for any later extracted subset.
- Do not commit the raw archive to Git.

## Commands

Run from the repository root.

1. Download the archive and verify the publisher MD5; compute local SHA-256:

   `python scripts/acquire_inria_pla.py acquire`

   An interrupted transfer leaves a `.part` file and can resume when the server supports HTTP Range requests. If a complete archive exists, the command verifies it rather than downloading again. Use `--force` to intentionally reacquire.

2. Verify the archive again:

   `python scripts/acquire_inria_pla.py verify`

3. Validate the ZIP CRCs and write a sorted inventory with per-member SHA-256 values, without extracting files:

   `python scripts/acquire_inria_pla.py inventory`

4. Run unit tests:

   `python -m unittest discover -s tests -p "test_dataset_acquisition.py" -v`

Generated local outputs:

- `data/external/inria_pla/PLA_dataset.zip`
- `data/external/inria_pla/acquisition_report.json`
- `data/external/inria_pla/archive_inventory.json`

These outputs are intentionally ignored by Git. Keep the original archive in a durable, access-controlled location permitted by the dataset's terms. Transfer the archive to another machine when useful; the other machine can also download it again from the lock file's pinned version URL.

## Integrity semantics

- Publisher MD5 checks whether the downloaded archive matches the checksum published for this release.
- Locally computed SHA-256 gives our project a stronger reproducibility identifier.
- ZIP CRC checks detect corrupt members; per-member SHA-256 values identify the exact archive contents.
- A checksum is not a licence, provenance review, or scientific suitability decision.
- Inventory file extensions only identify candidates. Confirm actual sample representation and metadata from the archive documentation before interpreting samples as complex I/Q.
- The script does not extract archive paths, modify samples, condition signals, or choose evaluation splits.

## Freeze procedure after acquisition

1. Inspect `archive_inventory.json`, archive README/docs, metadata, licence and collection protocol.
2. Confirm sample format, sample rate, center frequency, device/session/trial identifiers and any confounders from the actual release.
3. Decide the experiment question and split boundary before looking at MIN-versus-baseline results.
4. Select whole recordings/bursts with grouping that prevents the same source session or burst leaking across train/test splits.
5. Create a reviewed `subset_manifest.json` listing exact member paths, sizes and SHA-256 values, plus split assignment and rationale.
6. Update the tracked lock file with the verified archive SHA-256 and expected byte size only after independent verification. Record the licence with a source link and explicit conclusion; do not infer it.
7. Commit only small metadata/manifests/reports suitable for redistribution. Do not commit the raw archive or extracted raw recordings.
8. Re-run verification and tests on the environment that will perform Stage 16B-2.

## Failure handling

- MD5 mismatch: do not use the archive; preserve the error output, remove the bad archive/partial file and reacquire.
- Network/DNS failure: use another network-enabled environment; do not weaken checksum checks.
- ZIP/CRC failure: treat the archive as invalid.
- Missing or unclear licence: pause research use/redistribution pending clarification.
- Metadata cannot support session-aware splits: document the limitation and revise the experimental design before freezing the subset.
