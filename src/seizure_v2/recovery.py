"""Compact, verified recovery snapshots; never includes source signals or caches."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

from seizure_v2.common import sha256

PREFIX = "EEG_RECOVERY "
LIMIT = 512 * 1024**2


def eligible(path):
    parts = PurePosixPath(path).parts
    if len(parts) == 1:
        return parts[0] == "study.json"
    if parts[0] in {"baseline", "residual"}:
        return len(parts) == 2 and parts[1] in {
            "best.pt", "last.pt", "scaler.json", "run.json", "history.json", "frozen.json"}
    if parts[0] == "manifests":
        return len(parts) == 2 and parts[1] in {
            "dataset.json", "split.json", "recordings.csv", "exclusions.csv", "windows.csv.gz"}
    return parts[0] in {"reports", "environments"} and Path(path).suffix in {
        ".json", ".csv", ".png", ".pdf", ".txt"}


def snapshot(root, destination, stage):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.is_relative_to(root):
        raise ValueError("Recovery archive must be outside the run directory")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not eligible(relative):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Recovery cannot contain symlinks")
        if path.is_file():
            files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    if sum(item["bytes"] for item in files.values()) > LIMIT:
        raise ValueError("Compact recovery exceeds 512 MiB; inspect before proceeding")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    manifest = {"schema": 1, "stage": stage, "files": files,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git.stdout.strip(),
                "restore": "Reprepare source data on runtime disk, restore this run, then resume. No EEG arrays are included."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    with tarfile.open(temp, "w:gz") as archive:
        for relative in files:
            archive.add(root / relative, arcname="run/" + relative, recursive=False)
        payload = json.dumps(manifest, sort_keys=True, indent=2).encode()
        member = tarfile.TarInfo("recovery.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    temp.replace(destination)
    return {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size,
            "stage": stage}


def inspect_archive(archive, digest):
    if sha256(archive) != digest:
        raise ValueError("Recovery archive SHA-256 mismatch")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or "recovery.json" not in names:
            raise ValueError("Invalid recovery members")
        if any(not m.isfile() or m.size < 0 for m in members) or sum(m.size for m in members) > LIMIT + 1024**2:
            raise ValueError("Unsafe or oversized recovery archive")
        manifest_member = source.getmember("recovery.json")
        if manifest_member.size > 1024**2:
            raise ValueError("Oversized recovery manifest")
        manifest = json.load(source.extractfile(manifest_member))
        if manifest["schema"] != 1 or set(names) != {"recovery.json", *("run/" + p for p in manifest["files"])}:
            raise ValueError("Recovery inventory mismatch")
        import hashlib
        for relative, expected in manifest["files"].items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or not eligible(relative):
                raise ValueError("Unsafe recovery path")
            member = source.getmember("run/" + relative)
            if member.size != expected["bytes"]:
                raise ValueError("Recovery size mismatch")
            digest_file = hashlib.sha256()
            with source.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    digest_file.update(block)
            if digest_file.hexdigest() != expected["sha256"]:
                raise ValueError("Recovery member hash mismatch")
    return manifest


def restore(archive, digest, output):
    manifest = inspect_archive(archive, digest)
    output = Path(output)
    if output.exists():
        raise ValueError("Restore requires a new output directory; existing runs are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "run"
        staged.mkdir()
        with tarfile.open(archive, "r:gz") as source:
            for relative in manifest["files"]:
                path = staged / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile("run/" + relative) as incoming, path.open("wb") as target:
                    shutil.copyfileobj(incoming, target)
        staged.rename(output)
    return manifest


def checkpoint(stage):
    """Pause at an atomic checkpoint until the local receiver verifies its download."""
    root = os.environ.get("EEG_RECOVERY_ROOT")
    if not root:
        return
    destination = Path(os.environ["EEG_RECOVERY_DIR"]) / "eeg-recovery.tar.gz"
    event = snapshot(root, destination, stage)
    print(PREFIX + json.dumps(event), flush=True)
    if sys.stdin.readline().strip() != event["sha256"]:
        raise RuntimeError("Recovery download was not acknowledged with the matching SHA-256; run remains resumable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["verify", "restore"])
    parser.add_argument("--archive", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.action == "restore":
        if not args.output:
            parser.error("restore requires --output")
        result = restore(args.archive, args.sha256, args.output)
    else:
        result = inspect_archive(args.archive, args.sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
