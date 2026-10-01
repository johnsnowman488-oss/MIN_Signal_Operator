"""Tests for Stage 16B-1 raw physical-IQ ingestion and provenance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "experiments_16b1",
    ROOT / "experiments" / "16B_1_iq_ingestion.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)

REQUIRED_METADATA = MOD.REQUIRED_METADATA
SAMPLE_REPRESENTATION = MOD.SAMPLE_REPRESENTATION
build_recording_manifest = MOD.build_recording_manifest
iq_integrity_summary = MOD.iq_integrity_summary
load_complex64_iq = MOD.load_complex64_iq
sha256_file = MOD.sha256_file
validate_metadata = MOD.validate_metadata


def fixture_metadata() -> dict:
    return {
        "recording_id": "test-recording-0001",
        "dataset_id": "fixture",
        "dataset_tier": 1,
        "source_doi": "10.5281/example",
        "source_version": "v1.0",
        "sample_rate_hz": 2_000_000.0,
        "center_frequency_hz": 400_000_000.0,
        "sample_representation": SAMPLE_REPRESENTATION,
        "source_checksum": "publisher-checksum-placeholder",
        "conditioning_version": "raw-v0",
        "subset_id": "subset-0001",
        "split_id": "train",
    }


def write_fixture(path: Path) -> np.ndarray:
    t = np.arange(8, dtype=np.float32)
    expected = (0.25 * np.sin(t)).astype(np.float32) + 1j * (
        -0.5 * np.cos(t / 2.0)
    ).astype(np.float32)

    wire = np.empty(expected.size * 2, dtype="<f4")
    wire[0::2] = expected.real
    wire[1::2] = expected.imag
    wire.tofile(path)
    return expected.astype(np.complex64)


def test_loads_interleaved_complex64_exactly(tmp_path: Path) -> None:
    path = tmp_path / "iq.bin"
    expected = write_fixture(path)
    loaded = load_complex64_iq(path)
    assert loaded.dtype == np.complex64
    assert np.array_equal(loaded, expected)


def test_sample_slice_uses_sample_indices(tmp_path: Path) -> None:
    path = tmp_path / "iq.bin"
    expected = write_fixture(path)
    loaded = load_complex64_iq(path, start=2, stop=5)
    assert np.array_equal(loaded, expected[2:5])


def test_rejects_incomplete_iq_pair(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    np.arange(5, dtype="<f4").tofile(path)
    with pytest.raises(ValueError, match="even number"):
        load_complex64_iq(path)


def test_metadata_contract_is_complete_and_valid() -> None:
    result = validate_metadata(fixture_metadata())
    assert result["valid"] is True
    assert result["missing"] == []
    assert result["errors"] == []
    assert all(key in fixture_metadata() for key in REQUIRED_METADATA)


def test_metadata_contract_rejects_bad_representation() -> None:
    metadata = fixture_metadata()
    metadata["sample_representation"] = "complex128"
    result = validate_metadata(metadata)
    assert result["valid"] is False
    assert any("sample_representation" in error for error in result["errors"])


def test_integrity_summary_detects_nonfinite_samples() -> None:
    iq = np.array([1 + 2j, np.nan + 0j, 0 + 1j * np.inf], dtype=np.complex64)
    result = iq_integrity_summary(iq)
    assert result["sample_count"] == 3
    assert result["finite_sample_count"] == 1
    assert result["nonfinite_sample_count"] == 2
    assert result["finite_fraction"] == pytest.approx(1 / 3)


def test_manifest_hash_is_hash_of_exact_source_bytes(tmp_path: Path) -> None:
    path = tmp_path / "iq.bin"
    write_fixture(path)
    manifest = build_recording_manifest(path, fixture_metadata())
    assert manifest["source"]["sha256"] == sha256_file(path)
    assert manifest["source"]["bytes"] == path.stat().st_size
    assert manifest["ingestion"]["samples_read"] == 8
    assert manifest["ingestion"]["truncated_read"] is False


def test_bounded_read_does_not_change_source_hash(tmp_path: Path) -> None:
    path = tmp_path / "iq.bin"
    write_fixture(path)
    full = build_recording_manifest(path, fixture_metadata())
    bounded = build_recording_manifest(path, fixture_metadata(), max_samples=3)
    assert bounded["source"]["sha256"] == full["source"]["sha256"]
    assert bounded["ingestion"]["samples_read"] == 3
    assert bounded["ingestion"]["truncated_read"] is True
    assert bounded["integrity"]["sample_count"] == 3


def test_integrity_summary_reports_iq_statistics() -> None:
    iq = np.array([1 + 0j, -1 + 0j, 0 + 1j, 0 - 1j], dtype=np.complex64)
    result = iq_integrity_summary(iq)
    assert result["mean_i"] == pytest.approx(0.0)
    assert result["mean_q"] == pytest.approx(0.0)
    assert result["rms_i"] == pytest.approx(1 / np.sqrt(2))
    assert result["rms_q"] == pytest.approx(1 / np.sqrt(2))
    assert result["iq_power_ratio_q_over_i"] == pytest.approx(1.0)
