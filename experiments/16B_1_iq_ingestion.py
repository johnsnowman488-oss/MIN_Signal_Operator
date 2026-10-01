"""Stage 16B-1: physical-IQ ingestion and provenance validation.

Scope:
    16A -> freeze dataset/source metadata and acquisition policy.
    16B-1 -> validate the byte-level IQ ingestion contract before any
             temporal characterization or MIN/baseline evaluation.

This module deliberately does not:
    * download external datasets;
    * condition, normalize, synchronize, or filter raw samples;
    * fit a memory model;
    * tune MIN geometry;
    * compute task/communications performance.

The primary supported physical-IQ wire format in the first Paper-1 datasets is
headerless complex64-equivalent data stored as interleaved float32 I,Q:
    [I0, Q0, I1, Q1, ...]

All hashes are SHA-256 over the exact source bytes consumed by the loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "1.0"
SAMPLE_REPRESENTATION = "complex64_interleaved_float32_iq"

REQUIRED_METADATA = (
    "recording_id",
    "dataset_id",
    "dataset_tier",
    "source_doi",
    "source_version",
    "sample_rate_hz",
    "center_frequency_hz",
    "sample_representation",
    "source_checksum",
    "conditioning_version",
    "subset_id",
    "split_id",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 for the exact bytes of path using bounded memory."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complex64_iq(
    path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Load headerless interleaved float32 I,Q into a complex64 array.

    start and stop are sample indices, not float32 indices. Loading is
    intentionally strict: an odd number of float32 values is rejected because
    it represents an incomplete I/Q pair.
    """
    if start < 0:
        raise ValueError("start must be non-negative")
    if stop is not None and stop < start:
        raise ValueError("stop must be greater than or equal to start")

    raw = np.fromfile(Path(path), dtype="<f4")
    if raw.size % 2:
        raise ValueError(
            f"invalid interleaved IQ byte stream: {raw.size} float32 values; "
            "expected an even number"
        )

    iq = raw[0::2].astype(np.complex64)
    iq.imag = raw[1::2]

    if stop is None:
        return iq[start:]
    return iq[start:stop]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def iq_integrity_summary(iq: np.ndarray) -> dict[str, Any]:
    """Return non-destructive integrity diagnostics for a complex IQ vector."""
    samples = np.asarray(iq, dtype=np.complex64)
    if samples.ndim != 1:
        raise ValueError("IQ input must be a one-dimensional complex array")

    finite_mask = np.isfinite(samples.real) & np.isfinite(samples.imag)
    finite_count = int(np.count_nonzero(finite_mask))
    sample_count = int(samples.size)

    real = samples.real.astype(np.float64, copy=False)
    imag = samples.imag.astype(np.float64, copy=False)

    real_power = float(np.mean(real[finite_mask] ** 2)) if finite_count else None
    imag_power = float(np.mean(imag[finite_mask] ** 2)) if finite_count else None
    magnitude = np.abs(samples[finite_mask]) if finite_count else np.asarray([], dtype=float)

    return {
        "sample_count": sample_count,
        "finite_sample_count": finite_count,
        "nonfinite_sample_count": sample_count - finite_count,
        "finite_fraction": float(finite_count / sample_count) if sample_count else 0.0,
        "mean_i": float(np.mean(real[finite_mask])) if finite_count else None,
        "mean_q": float(np.mean(imag[finite_mask])) if finite_count else None,
        "rms_i": float(np.sqrt(real_power)) if real_power is not None else None,
        "rms_q": float(np.sqrt(imag_power)) if imag_power is not None else None,
        "iq_power_ratio_q_over_i": _safe_ratio(
            imag_power or 0.0, real_power or 0.0
        ),
        "mean_power": float(np.mean(np.abs(samples[finite_mask]) ** 2))
        if finite_count
        else None,
        "peak_magnitude": float(np.max(magnitude)) if magnitude.size else None,
    }


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum provenance contract required by Stage 16B-1."""
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    errors: list[str] = []

    if missing:
        errors.append(f"missing required metadata fields: {missing}")

    if metadata.get("sample_representation") != SAMPLE_REPRESENTATION:
        errors.append(
            "sample_representation must be "
            f"{SAMPLE_REPRESENTATION!r} for this loader"
        )

    sample_rate = metadata.get("sample_rate_hz")
    if sample_rate is not None and float(sample_rate) <= 0:
        errors.append("sample_rate_hz must be positive")

    center_frequency = metadata.get("center_frequency_hz")
    if center_frequency is not None and float(center_frequency) < 0:
        errors.append("center_frequency_hz must be non-negative")

    tier = metadata.get("dataset_tier")
    if tier is not None and int(tier) not in (0, 1, 2):
        errors.append("dataset_tier must be 0, 1, or 2")

    return {
        "valid": not errors,
        "missing": missing,
        "errors": errors,
    }


def build_recording_manifest(
    path: str | Path,
    metadata: Mapping[str, Any],
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Load, validate, and describe one raw recording without modifying it."""
    metadata_result = validate_metadata(metadata)
    if not metadata_result["valid"]:
        raise ValueError("; ".join(metadata_result["errors"]))

    sample_path = Path(path)
    iq = load_complex64_iq(sample_path, stop=max_samples)
    source_sha256 = sha256_file(sample_path)
    integrity = iq_integrity_summary(iq)

    sample_rate_hz = float(metadata["sample_rate_hz"])
    duration_seconds = integrity["sample_count"] / sample_rate_hz

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "16B-1",
        "recording": {
            "recording_id": str(metadata["recording_id"]),
            "dataset_id": str(metadata["dataset_id"]),
            "dataset_tier": int(metadata["dataset_tier"]),
            "source_doi": str(metadata["source_doi"]),
            "source_version": str(metadata["source_version"]),
            "sample_representation": str(metadata["sample_representation"]),
            "sample_rate_hz": sample_rate_hz,
            "center_frequency_hz": float(metadata["center_frequency_hz"]),
            "source_checksum": str(metadata["source_checksum"]),
            "conditioning_version": str(metadata["conditioning_version"]),
            "subset_id": str(metadata["subset_id"]),
            "split_id": str(metadata["split_id"]),
        },
        "source": {
            "path": str(sample_path),
            "sha256": source_sha256,
            "bytes": sample_path.stat().st_size,
        },
        "ingestion": {
            "loader": "load_complex64_iq",
            "wire_dtype": "little-endian float32",
            "wire_layout": "interleaved I,Q",
            "samples_read": integrity["sample_count"],
            "duration_seconds": float(duration_seconds),
            "truncated_read": max_samples is not None,
        },
        "integrity": integrity,
    }


def _demo_metadata() -> dict[str, Any]:
    return {
        "recording_id": "16B-1-demo-0001",
        "dataset_id": "fixture",
        "dataset_tier": 0,
        "source_doi": "fixture://16B-1",
        "source_version": "fixture-v1",
        "sample_rate_hz": 1_000_000.0,
        "center_frequency_hz": 2_400_000_000.0,
        "sample_representation": SAMPLE_REPRESENTATION,
        "source_checksum": "fixture-source-checksum-not-a-publisher-hash",
        "conditioning_version": "raw-v0",
        "subset_id": "fixture-subset-0001",
        "split_id": "fixture-test",
    }


def self_test() -> dict[str, Any]:
    """Exercise the complete byte-to-IQ-to-manifest path on a deterministic fixture."""
    with tempfile.TemporaryDirectory(prefix="min-16b1-") as tmpdir:
        raw_path = Path(tmpdir) / "fixture.bin"

        t = np.arange(32, dtype=np.float32)
        i = np.sin(t / 5.0).astype(np.float32)
        q = np.cos(t / 7.0).astype(np.float32)
        interleaved = np.empty(t.size * 2, dtype="<f4")
        interleaved[0::2] = i
        interleaved[1::2] = q
        interleaved.tofile(raw_path)

        metadata = _demo_metadata()
        manifest = build_recording_manifest(raw_path, metadata)

        assert manifest["integrity"]["sample_count"] == 32
        assert manifest["integrity"]["nonfinite_sample_count"] == 0
        assert manifest["source"]["sha256"] == sha256_file(raw_path)
        assert manifest["ingestion"]["wire_layout"] == "interleaved I,Q"
        return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="headerless interleaved I,Q binary")
    parser.add_argument("--metadata", type=Path, help="JSON provenance metadata")
    parser.add_argument("--output", type=Path, help="write manifest JSON to this path")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional sample cap for a bounded smoke test",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the deterministic ingestion fixture and print its manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.self_test:
        manifest = self_test()
    else:
        if args.input is None or args.metadata is None:
            raise SystemExit(
                "--input and --metadata are required unless --self-test is used"
            )
        metadata = json.loads(args.metadata.read_text())
        manifest = build_recording_manifest(
            args.input,
            metadata,
            max_samples=args.max_samples,
        )

    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
