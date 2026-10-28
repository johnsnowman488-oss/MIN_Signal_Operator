#!/usr/bin/env python3
"""Reproducible INRIA PLA archive acquisition and inventory (standard library only).

Commands:
  acquire   Download/resume archive, verify publisher MD5, record local SHA-256.
  inventory Verify archive and write a deterministic member inventory.
  verify    Recheck archive against the local acquisition report.
No archive is extracted automatically; this avoids unsafe paths and accidental data commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "datasets" / "inria_pla" / "dataset.lock.json"
DATA_DIR = ROOT / "data" / "external" / "inria_pla"
ARCHIVE_PATH = DATA_DIR / "PLA_dataset.zip"
REPORT_PATH = DATA_DIR / "acquisition_report.json"
INVENTORY_PATH = DATA_DIR / "archive_inventory.json"
USER_AGENT = "MIN-Signal-Operator-Stage16B1/1.0"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required only to compare publisher's MD5.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def get_config() -> dict[str, Any]:
    config = load_json(LOCK_PATH)
    if config.get("dataset_id") != "inria_iq_rf_fingerprinting":
        raise ValueError("Unexpected dataset_id in lock file")
    if not config.get("download_url", "").startswith("https://"):
        raise ValueError("Lock file must contain an HTTPS download_url")
    return config


def download_with_resume(url: str, destination: Path, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    # Server ignored Range or restarted transfer: restart safely.
                    offset = 0
                    mode = "wb"
                else:
                    mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            partial.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt + 1 >= retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {exc}") from exc
            print(f"Download attempt {attempt + 1}/{retries} failed: {exc}; retrying.", file=sys.stderr)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Download failed")


def verify_archive(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Archive not found: {path}. Run the acquire command first.")
    expected = config["publisher_checksum"]["value"].lower()
    actual_md5 = md5_file(path)
    if actual_md5.lower() != expected:
        raise ValueError(
            f"Publisher MD5 mismatch: expected {expected}, got {actual_md5}. "
            "Keep the file for diagnosis but do not use it; remove it and reacquire."
        )
    if not zipfile.is_zipfile(path):
        raise ValueError("Checksum matched, but the downloaded file is not a valid ZIP archive.")
    return {
        "archive_filename": path.name,
        "archive_bytes": path.stat().st_size,
        "publisher_checksum": {"algorithm": "md5", "expected": expected, "actual": actual_md5, "verified": True},
        "local_sha256": sha256_file(path),
        "zip_container_valid": True,
    }


def safe_member_name(name: str) -> bool:
    # ZIP member names use POSIX separators even on Windows.
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and not (p.parts and ":" in p.parts[0])


def build_inventory(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC validation failed for member: {bad_member}")
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            row: dict[str, Any] = {
                "path": info.filename,
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "is_directory": info.is_dir(),
                "safe_relative_path": safe_member_name(info.filename),
            }
            if not info.is_dir() and safe_member_name(info.filename):
                digest = hashlib.sha256()
                with archive.open(info, "r") as member:
                    for chunk in iter(lambda: member.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                row["sha256"] = digest.hexdigest()
            rows.append(row)
    return {
        "archive": path.name,
        "archive_sha256": sha256_file(path),
        "member_count": len(rows),
        "unsafe_member_paths": [row["path"] for row in rows if not row["safe_relative_path"]],
        "members": rows,
        "raw_iq_candidates": [
            row for row in rows
            if not row["is_directory"] and row["safe_relative_path"]
            and PurePosixPath(row["path"]).suffix.lower() in {".bin", ".iq", ".dat", ".raw", ".cf32"}
        ],
        "note": "Inventory only. Candidate extensions are not proof of I/Q format; inspect dataset documentation and metadata.",
    }


def command_acquire(args: argparse.Namespace) -> None:
    config = get_config()
    if args.archive:
        destination = Path(args.archive).expanduser().resolve()
    else:
        destination = ARCHIVE_PATH
    if destination.exists() and not args.force:
        print(f"Archive already exists: {destination}")
    else:
        download_with_resume(config["download_url"], destination)
    result = verify_archive(destination, config)
    result.update({
        "dataset_id": config["dataset_id"],
        "doi": config["doi"],
        "release": config["release"],
        "source_page": config["source_page"],
        "download_url": config["download_url"],
        "acquired_archive_path": str(destination),
        "status": "archive_verified",
        "raw_archive_committed_to_git": False,
    })
    report_path = REPORT_PATH if destination == ARCHIVE_PATH else destination.with_name("acquisition_report.json")
    write_json(report_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nVerified. Report: {report_path}")


def command_verify(args: argparse.Namespace) -> None:
    config = get_config()
    path = Path(args.archive).expanduser().resolve() if args.archive else ARCHIVE_PATH
    result = verify_archive(path, config)
    report_path = REPORT_PATH if path == ARCHIVE_PATH else path.with_name("acquisition_report.json")
    if report_path.exists():
        prior = load_json(report_path)
        if prior.get("local_sha256") != result["local_sha256"]:
            raise ValueError("Archive SHA-256 differs from the recorded acquisition report.")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Verification passed.")


def command_inventory(args: argparse.Namespace) -> None:
    config = get_config()
    path = Path(args.archive).expanduser().resolve() if args.archive else ARCHIVE_PATH
    verified = verify_archive(path, config)
    inventory = build_inventory(path)
    inventory.update({"dataset_id": config["dataset_id"], "doi": config["doi"], "release": config["release"]})
    if inventory["unsafe_member_paths"]:
        print("WARNING: unsafe ZIP paths detected; no extraction is performed.", file=sys.stderr)
    output = Path(args.output).expanduser().resolve() if args.output else (
        INVENTORY_PATH if path == ARCHIVE_PATH else path.with_name("archive_inventory.json")
    )
    write_json(output, inventory)
    print(json.dumps({
        "archive_sha256": verified["local_sha256"],
        "member_count": inventory["member_count"],
        "raw_iq_candidate_count": len(inventory["raw_iq_candidates"]),
        "unsafe_member_count": len(inventory["unsafe_member_paths"]),
        "inventory_path": str(output),
    }, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire", help="Download (with resume), then verify publisher MD5 and local SHA-256.")
    acquire.add_argument("--archive", help="Optional archive destination; defaults to data/external/inria_pla/PLA_dataset.zip")
    acquire.add_argument("--force", action="store_true", help="Re-download even if the destination already exists.")
    acquire.set_defaults(func=command_acquire)
    verify = sub.add_parser("verify", help="Recheck archive integrity and any existing acquisition report.")
    verify.add_argument("--archive", help="Optional archive path.")
    verify.set_defaults(func=command_verify)
    inventory = sub.add_parser("inventory", help="Verify ZIP CRCs and hash each safe archive member without extraction.")
    inventory.add_argument("--archive", help="Optional archive path.")
    inventory.add_argument("--output", help="Optional output JSON path.")
    inventory.set_defaults(func=command_inventory)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
