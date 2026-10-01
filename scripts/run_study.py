"""Run the frozen full-cohort experiment. Safe to restart after a Colab timeout."""
import argparse
import gzip
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from seizure_v2.recovery import checkpoint
from seizure_v2.common import read_json
from seizure_v2.data.edf import cache_paths


def run(*args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interactive-backups", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true", help="Use an intact runtime cache after a full integrity audit; fail if missing or incomplete")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    study_lock = (output / ".study.lock").open("a")
    try:
        fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another study process is already using this output directory") from error
    if args.interactive_backups:
        os.environ["EEG_RECOVERY_ROOT"] = str(output.resolve())
        os.environ["EEG_RECOVERY_DIR"] = str(output.resolve().parent / "recovery-staging")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 (Colab runtime version 2026.07)")
    environment = output / "environments" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    environment.mkdir(parents=True)
    (environment / "requirements.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    (environment / "hardware.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    cli = [sys.executable, "-m", "seizure_v2.cli"]
    if not args.reuse_prepared:
        run(*cli, "prepare", "--root", args.data, "--full", "--evict-raw", "--mirror", "s3", "--workers", "4")
    run(*cli, "audit", "--root", args.data, "--require-full")
    # Persist provenance, not the signal arrays or downloaded EDF files.
    manifests = output / "manifests"
    manifests.mkdir(exist_ok=True)
    for name in ["dataset.json", "split.json", "recordings.csv", "exclusions.csv"]:
        shutil.copyfile(Path(args.data) / "manifests" / name, manifests / name)
    dataset = read_json(manifests / "dataset.json")
    for entry in dataset["prepared"] + dataset["exclusions"]:
        metadata = cache_paths(args.data, entry["recording"])[1]
        target = manifests / "recording-metadata" / Path(entry["recording"]).with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata, target)
    source_names = ["SHA256SUMS.txt", "RECORDS", "RECORDS-WITH-SEIZURES"] + [
        f"chb{i:02d}/chb{i:02d}-summary.txt" for i in range(1, 25)]
    for name in source_names:
        target = manifests / "source-metadata" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(args.data) / "source_metadata" / name, target)
    with (Path(args.data) / "manifests/windows.csv").open("rb") as source:
        with gzip.open(manifests / "windows.csv.gz", "wb") as destination:
            shutil.copyfileobj(source, destination)
    checkpoint("full-cohort-prepared-and-audited")
    for name, config in [("baseline", "configs/baseline.yaml"), ("residual", "configs/improved.yaml")]:
        destination = output / name
        if not (destination / "frozen.json").exists():
            command = [*cli, "train", "--root", args.data, "--config", config, "--output", destination, "--device", "cuda"]
            if (destination / "last.pt").exists():
                command.append("--resume")
            run(*command)
    study = output / "study.json"
    if not study.exists():
        run(*cli, "freeze", "--runs", output / "baseline", output / "residual", "--output", study)
    checkpoint("study-frozen-before-test")
    run(*cli, "evaluate", "--root", args.data, "--study", study, "--output", output / "reports")
    checkpoint("held-out-evaluation-complete")
    # Expanded example clips and packaging intermediates stay on runtime disk.
    bundle = Path(args.data) / "release/demo"
    if not (bundle / "manifest.json").exists():
        run(*cli, "bundle", "--root", args.data, "--study", study, "--reports", output / "reports", "--output", bundle)
    shutil.copyfile(bundle.with_suffix(".tar.gz"), output / "demo.tar.gz")
    checkpoint("release-complete")
    print(f"Complete. Reports: {output / 'reports'}; release: {output / 'demo.tar.gz'}", flush=True)


if __name__ == "__main__":
    main()
