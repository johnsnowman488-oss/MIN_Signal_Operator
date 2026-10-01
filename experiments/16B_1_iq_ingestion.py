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

All hashes cover the exact source bytes and are computed in bounded memory.
Integrity scanning is also chunked so large recordings are never materialized
as one in-memory array.
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
BYTES_PER_SAMPLE = 8
DEFAULT_HASH_CHUNK_BYTES = 1024 * 1024
DEFAULT_IQ_CHUNK_SAMPLES = 1024 * 1024

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


def sha256_file(
    path: str | Path,
    chunk_size: int = DEFAULT_HASH_CHUNK_BYTES,
) -> str:
    """Return SHA-256 for the exact bytes of path using bounded memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_iq_sample_count(path: str | Path) -> int:
    """Return the complete sample count for interleaved float32 I,Q bytes."""
    file_path = Path(path)
    size = file_path.stat().st_size
    if size % BYTES_PER_SAMPLE:
        raise ValueError(
            f"invalid IQ byte stream: {size} bytes; expected a multiple of "
            f"{BYTES_PER_SAMPLE} bytes per complex sample"
        )
    return size // BYTES_PER_SAMPLE


def load_complex64_iq(
    path: str | Path,
    *,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Load only the requested sample interval from a raw IQ binary file.

    start and stop are sample indices. The loader seeks directly to the
    requested byte offset instead of reading the complete source file.
    """
    if start < 0:
        raise ValueError("start must be non-negative")
    if stop is not None and stop < start:
        raise ValueError("stop must be greater than or equal to start")

    total_samples = raw_iq_sample_count(path)
    if start > total_samples:
        raise ValueError(
            f"start={start} exceeds available sample count {total_samples}"
        )
    if stop is None:
        stop = total_samples
    if stop > total_samples:
        raise ValueError(
            f"stop={stop} exceeds available sample count {total_samples}"
        )

    count = stop - start
    if count == 0:
        return np.empty(0, dtype=np.complex64)

    file_path = Path(path)
    with file_path.open("rb") as handle:
        handle.seek(start * BYTES_PER_SAMPLE)
        raw = np.fromfile(handle, dtype="<f4", count=2 * count)

    if raw.size != 2 * count:
        raise ValueError(
            f"short IQ read: expected {2 * count} float32 values, got {raw.size}"
        )

    iq = raw[0::2].astype(np.complex64)
    iq.imag = raw[1::2]
    return iq


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _integrity_summary_from_accumulators(
    *,
    sample_count: int,
    finite_count: int,
    sum_i: float,
    sum_q: float,
    sum_i2: float,
    sum_q2: float,
    sum_power: float,
    peak_magnitude: float | None,
) -> dict[str, Any]:
    if finite_count:
        mean_i = sum_i / finite_count
        mean_q = sum_q / finite_count
        real_power = sum_i2 / finite_count
        imag_power = sum_q2 / finite_count
        mean_power = sum_power / finite_count
    else:
        mean_i = mean_q = real_power = imag_power = mean_power = None

    return {
        "sample_count": sample_count,
        "finite_sample_count": finite_count,
        "nonfinite_sample_count": sample_count - finite_count,
        "finite_fraction": float(finite_count / sample_count)
        if sample_count
        else 0.0,
        "mean_i": float(mean_i) if mean_i is not None else None,
        "mean_q": float(mean_q) if mean_q is not None else None,
        "rms_i": float(np.sqrt(real_power)) if real_power is not None else None,
        "rms_q": float(np.sqrt(imag_power)) if imag_power is not None else None,
        "iq_power_ratio_q_over_i": _safe_ratio(
            imag_power or 0.0,
            real_power or 0.0,
        ),
        "mean_power": float(mean_power) if mean_power is not None else None,
        "peak_magnitude": float(peak_magnitude)
        if peak_magnitude is not None
        else None,
    }


def iq_integrity_summary(iq: np.ndarray) -> dict[str, Any]:
    """Return non-destructive integrity diagnostics for a complex IQ vector."""
    samples = np.asarray(iq, dtype=np.complex64)
    if samples.ndim != 1:
        raise ValueError("IQ input must be a one-dimensional complex array")

    finite_mask = np.isfinite(samples.real) & np.isfinite(samples.imag)
    finite = samples[finite_mask]
    real = finite.real.astype(np.float64, copy=False)
    imag = finite.imag.astype(np.float64, copy=False)

    return _integrity_summary_from_accumulators(
        sample_count=int(samples.size),
        finite_count=int(finite.size),
        sum_i=float(np.sum(real)),
        sum_q=float(np.sum(imag)),
        sum_i2=float(np.sum(real**2)),
        sum_q2=float(np.sum(imag**2)),
        sum_power=float(np.sum(real**2 + imag**2)),
        peak_magnitude=float(np.max(np.abs(finite)))
        if finite.size
        else None,
    )


def iq_file_integrity_summary(
    path: str | Path,
    *,
    max_samples: int | None = None,
    chunk_samples: int = DEFAULT_IQ_CHUNK_SAMPLES,
) -> dict[str, Any]:
    """Scan a raw IQ file in bounded chunks without materializing the file."""
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")

    file_path = Path(path)
    total_samples = raw_iq_sample_count(file_path)
    sample_limit = total_samples if max_samples is None else min(
        max_samples, total_samples
    )

    sample_count = 0
    finite_count = 0
    sum_i = sum_q = sum_i2 = sum_q2 = sum_power = 0.0
    peak_magnitude: float | None = None

    with file_path.open("rb") as handle:
        while sample_count < sample_limit:
            request = min(chunk_samples, sample_limit - sample_count)
            raw = np.fromfile(handle, dtype="<f4", count=2 * request)
            if raw.size != 2 * request:
                raise ValueError(
                    f"short IQ chunk: expected {2 * request} float32 values, "
                    f"got {raw.size}"
                )

            real = raw[0::2].astype(np.float64, copy=False)
            imag = raw[1::2].astype(np.float64, copy=False)
            finite_mask = np.isfinite(real) & np.isfinite(imag)
            finite_real = real[finite_mask]
            finite_imag = imag[finite_mask]

            sample_count += request
            finite_count += int(finite_real.size)
            if finite_real.size:
                sum_i += float(np.sum(finite_real))
                sum_q += float(np.sum(finite_imag))
                sum_i2 += float(np.sum(finite_real**2))
                sum_q2 += float(np.sum(finite_imag**2))
                sum_power += float(
                    np.sum(finite_real**2 + finite_imag**2)
                )
                chunk_peak = float(
                    np.max(np.sqrt(finite_real**2 + finite_imag**2))
                )
                peak_magnitude = (
                    chunk_peak
                    if peak_magnitude is None
                    else max(peak_magnitude, chunk_peak)
                )

    return _integrity_summary_from_accumulators(
        sample_count=sample_count,
        finite_count=finite_count,
        sum_i=sum_i,
        sum_q=sum_q,
        sum_i2=sum_i2,
        sum_q2=sum_q2,
        sum_power=sum_power,
        peak_magnitude=peak_magnitude,
    )


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
    chunk_samples: int = DEFAULT_IQ_CHUNK_SAMPLES,
) -> dict[str, Any]:
    """Validate and describe one raw recording without modifying it."""
    metadata_result = validate_metadata(metadata)
    if not metadata_result["valid"]:
        raise ValueError("; ".join(metadata_result["errors"]))

    sample_path = Path(path)
    total_samples = raw_iq_sample_count(sample_path)
    integrity = iq_file_integrity_summary(
        sample_path,
        max_samples=max_samples,
        chunk_samples=chunk_samples,
    )
    source_sha256 = sha256_file(sample_path)

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
            "samples_total": total_samples,
            "samples_read": integrity["sample_count"],
            "duration_seconds": float(duration_seconds),
            "truncated_read": integrity["sample_count"] < total_samples,
            "chunk_samples": int(chunk_samples),
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
    """Exercise the byte-to-IQ-to-manifest path on a deterministic fixture."""
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
        manifest = build_recording_manifest(
            raw_path,
            metadata,
            chunk_samples=7,
        )

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
        "--chunk-samples",
        type=int,
        default=DEFAULT_IQ_CHUNK_SAMPLES,
        help="integrity-scan chunk size in complex samples",
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
            chunk_samples=args.chunk_samples,
        )

    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
