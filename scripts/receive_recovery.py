"""Verify a browser download and retain the two latest project-owned snapshots."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
from seizure_v2.common import read_json, sha256, write_json
from seizure_v2.recovery import inspect_archive


def receive(source, digest, destination, move=False):
    source, destination = Path(source), Path(destination)
    manifest = inspect_archive(source, digest)
    destination.mkdir(parents=True, exist_ok=True)
    incoming = destination / "incoming.tar.gz"
    shutil.copyfile(source, incoming)
    if sha256(incoming) != digest:
        raise ValueError("Local recovery copy hash mismatch")
    latest, previous = destination / "latest.tar.gz", destination / "previous.tar.gz"
    receipt_path = destination / "receipts.json"
    receipts = read_json(receipt_path) if receipt_path.exists() else []
    # Never replace unknown files. Only rotate downloads recorded by this receiver.
    known = {item["sha256"] for item in receipts}
    for path in [latest, previous]:
        if path.exists() and sha256(path) not in known:
            raise ValueError(f"Refusing to overwrite an unrecorded recovery file: {path}")
    if latest.exists():
        latest.replace(previous)
    incoming.replace(latest)
    receipts.append({"sha256": digest, "stage": manifest["stage"],
                     "git_commit": manifest["git_commit"], "bytes": latest.stat().st_size,
                     "verified_utc": datetime.now(timezone.utc).isoformat()})
    write_json(receipt_path, receipts)
    if move and source.resolve() not in {latest.resolve(), previous.resolve()}:
        # Remove only the explicitly supplied, hash-verified download, after its
        # replacement and durable receipt exist. This avoids duplicate epoch files.
        source.unlink()
    return latest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", default="recovery/full-seed42")
    parser.add_argument("--move", action="store_true", help="Move the verified browser download into the recovery folder")
    args = parser.parse_args()
    print(receive(args.source, args.sha256, args.destination, move=args.move))
    print("Verified SHA-256:", args.sha256)
