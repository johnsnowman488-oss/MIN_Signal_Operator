import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "acquire_inria_pla.py"
SPEC = importlib.util.spec_from_file_location("acquire_inria_pla", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class AcquisitionHelpersTests(unittest.TestCase):
    def test_hash_helpers_match_known_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            payload = b"MIN acquisition test\x00\x01"
            path.write_bytes(payload)
            self.assertEqual(MODULE.sha256_file(path), hashlib.sha256(payload).hexdigest())
            self.assertEqual(MODULE.md5_file(path), hashlib.md5(payload).hexdigest())

    def test_inventory_is_sorted_and_hashes_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.zip"
            first = b"interleaved iq sample bytes"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("raw/device2_trial1.bin", first)
                archive.writestr("README.txt", "dataset fixture")
            inventory = MODULE.build_inventory(path)
            self.assertEqual([m["path"] for m in inventory["members"]], ["README.txt", "raw/device2_trial1.bin"])
            candidates = inventory["raw_iq_candidates"]
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["sha256"], hashlib.sha256(first).hexdigest())
            self.assertEqual(inventory["unsafe_member_paths"], [])

    def test_inventory_flags_unsafe_member_without_extracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escape.bin", b"not extracted")
            inventory = MODULE.build_inventory(path)
            self.assertEqual(inventory["unsafe_member_paths"], ["../escape.bin"])
            self.assertFalse(inventory["members"][0]["safe_relative_path"])

    def test_lock_file_has_expected_identity_and_explicit_unfrozen_hash(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "datasets/inria_pla/dataset.lock.json").read_text())
        self.assertEqual(config["doi"], "10.5281/zenodo.18268648")
        self.assertEqual(config["publisher_checksum"]["value"], "aff583bee6f4efccd08fe78c731bf03d")
        self.assertIsNone(config["local_sha256"])
        self.assertNotEqual(config["licence_status"], "confirmed")


if __name__ == "__main__":
    unittest.main()
