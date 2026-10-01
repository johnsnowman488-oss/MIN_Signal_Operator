from pathlib import Path
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "acq16b1", ROOT / "experiments" / "16B_1_acquisition.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_self_test_is_deterministic():
    first = MOD.self_test()
    second = MOD.self_test()
    assert first["source"]["local_sha256"] == second["source"]["local_sha256"]
    assert first["subset"]["selected_members"] == second["subset"]["selected_members"]


def test_selection_is_independent_of_archive_order():
    members = [
        {"name": "z.bin", "bytes": 1, "compressed_bytes": 1, "is_directory": False, "crc32": "0"},
        {"name": "a.bin", "bytes": 1, "compressed_bytes": 1, "is_directory": False, "crc32": "0"},
        {"name": "middle.txt", "bytes": 1, "compressed_bytes": 1, "is_directory": False, "crc32": "0"},
    ]
    forward = MOD.deterministic_select(members, seed=1601, max_members=2)
    reverse = MOD.deterministic_select(list(reversed(members)), seed=1601, max_members=2)
    assert [x["name"] for x in forward] == [x["name"] for x in reverse]
    assert len(forward) == 2


def test_checksum_verification_rejects_wrong_digest(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="mismatch"):
        MOD.verify_checksum(path, "md5", "00000000000000000000000000000000")
