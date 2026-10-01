"""Run the frozen full-cohort experiment. Safe to restart after a Colab timeout."""
import argparse
import gzip
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys


def run(*args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 (Colab runtime version 2026.07)")
    environment = output / "environments" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    environment.mkdir(parents=True)
    (environment / "requirements.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    (environment / "hardware.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    cli = [sys.executable, "-m", "seizure_v2.cli"]
    run(*cli, "prepare", "--root", args.data, "--full", "--evict-raw", "--mirror", "s3", "--workers", "4")
    run(*cli, "audit", "--root", args.data, "--require-full")
    # Persist provenance, not the signal arrays or downloaded EDF files.
    manifests = output / "manifests"
    manifests.mkdir(exist_ok=True)
    for name in ["dataset.json", "split.json"]:
        shutil.copyfile(Path(args.data) / "manifests" / name, manifests / name)
    with (Path(args.data) / "manifests/windows.csv").open("rb") as source:
        with gzip.open(manifests / "windows.csv.gz", "wb") as destination:
            shutil.copyfileobj(source, destination)
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
    run(*cli, "evaluate", "--root", args.data, "--study", study, "--output", output / "reports")
    # Expanded example clips and packaging intermediates stay on runtime disk.
    bundle = Path(args.data) / "release/demo"
    if not (bundle / "manifest.json").exists():
        run(*cli, "bundle", "--root", args.data, "--study", study, "--reports", output / "reports", "--output", bundle)
    shutil.copyfile(bundle.with_suffix(".tar.gz"), output / "demo.tar.gz")
    print(f"Complete. Reports: {output / 'reports'}; release: {output / 'demo.tar.gz'}", flush=True)


if __name__ == "__main__":
    main()
