"""Stage 16B acquisition and subset-freeze utilities.

This module does not download data during CI. It provides the reproducible
local acquisition contract used by the physical-IQ stage:

1. download an explicitly frozen source URL;
2. verify the publisher checksum before extraction;
3. compute a local SHA-256 over the exact acquired archive;
4. inventory archive members without modifying raw bytes;
5. select members deterministically from an explicit rule;
6. emit an acquisition/subset manifest that becomes immutable experimental
   provenance.

No raw data is committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

CHUNK_BYTES = 8 * 1024 * 1024


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MIN-Signal-Operator/16B-acquisition"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out, length=CHUNK_BYTES)


def verify_checksum(path: Path, algorithm: str, expected: str) -> str:
    actual = hash_file(path, algorithm)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{algorithm.upper()} mismatch for {path.name}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def inventory_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return [
            {
                "name": info.filename,
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "is_directory": info.is_dir(),
                "crc32": f"{info.CRC:08x}",
            }
            for info in archive.infolist()
        ]


def deterministic_select(
    members: list[dict[str, Any]],
    *,
    seed: int,
    suffixes: tuple[str, ...] = (".bin", ".h5", ".pkl", ".pickle"),
    max_members: int | None = None,
) -> list[dict[str, Any]]:
    """Stable hash-ranked selection; never depends on archive ordering."""
    candidates = [
        item for item in members
        if not item["is_directory"]
        and item["name"].lower().endswith(suffixes)
    ]
    ranked = sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item['name']}".encode("utf-8")
        ).hexdigest(),
    )
    if max_members is not None:
        ranked = ranked[:max_members]
    return ranked


def build_acquisition_manifest(
    *,
    dataset_id: str,
    source_url: str,
    source_version: str,
    publisher_checksum: dict[str, str],
    archive_path: Path,
    inventory: list[dict[str, Any]],
    selected_members: list[dict[str, Any]],
    subset_rule: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage": "16B-1",
        "purpose": "physical_data_acquisition_and_subset_freeze",
        "dataset_id": dataset_id,
        "source": {
            "url": source_url,
            "version": source_version,
            "archive_name": archive_path.name,
            "publisher_checksum": publisher_checksum,
            "local_sha256": hash_file(archive_path, "sha256"),
            "bytes": archive_path.stat().st_size,
        },
        "subset": {
            "rule": subset_rule,
            "seed": seed,
            "selection_is_order_independent": True,
            "inventory_member_count": len(inventory),
            "selected_member_count": len(selected_members),
            "selected_members": selected_members,
        },
        "raw_data_policy": {
            "stored_in_git": False,
            "source_bytes_modified": False,
        },
    }


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="min-16b-acq-") as tmp:
        root = Path(tmp)
        archive = root / "fixture.zip"
        members = ["b.bin", "a.bin", "notes.txt", "nested/c.bin"]
        with zipfile.ZipFile(archive, "w") as zf:
            for name in members:
                zf.writestr(name, f"fixture:{name}\n")

        inventory = inventory_zip(archive)
        selected_a = deterministic_select(inventory, seed=1601, max_members=2)
        selected_b = deterministic_select(list(reversed(inventory)), seed=1601, max_members=2)
        assert [x["name"] for x in selected_a] == [x["name"] for x in selected_b]

        manifest = build_acquisition_manifest(
            dataset_id="fixture",
            source_url="fixture://16B-1",
            source_version="fixture-v1",
            publisher_checksum={"algorithm": "sha256", "value": hash_file(archive, "sha256")},
            archive_path=archive,
            inventory=inventory,
            selected_members=selected_a,
            subset_rule="stable SHA-256 rank over eligible members",
            seed=1601,
        )
        assert manifest["source"]["local_sha256"] == hash_file(archive, "sha256")
        assert manifest["subset"]["selected_member_count"] == 2
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--url")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--version")
    parser.add_argument("--publisher-md5")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=1601)
    parser.add_argument("--max-members", type=int, default=None)
    args = parser.parse_args()

    if args.self_test:
        manifest = self_test()
    else:
        required = (args.url, args.dataset_id, args.version, args.publisher_md5, args.archive)
        if any(value is None for value in required):
            raise SystemExit(
                "--url, --dataset-id, --version, --publisher-md5 and --archive "
                "are required unless --self-test is used"
            )
        if args.download:
            download_file(args.url, args.archive)

        verify_checksum(args.archive, "md5", args.publisher_md5)
        inventory = inventory_zip(args.archive)
        selected = deterministic_select(
            inventory,
            seed=args.seed,
            max_members=args.max_members,
        )
        manifest = build_acquisition_manifest(
            dataset_id=args.dataset_id,
            source_url=args.url,
            source_version=args.version,
            publisher_checksum={"algorithm": "md5", "value": args.publisher_md5},
            archive_path=args.archive,
            inventory=inventory,
            selected_members=selected,
            subset_rule=(
                "Stable SHA-256 ranking of eligible raw signal members using "
                "the committed seed; exact selected member paths are frozen "
                "in the emitted manifest."
            ),
            seed=args.seed,
        )

    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
