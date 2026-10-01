"""Run the frozen full-cohort experiment. Safe to restart after a Colab timeout."""
import argparse
from pathlib import Path
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
    cli = [sys.executable, "-m", "seizure_v2.cli"]
    run(*cli, "prepare", "--root", args.data, "--full", "--evict-raw", "--mirror", "s3", "--workers", "4")
    run(*cli, "audit", "--root", args.data, "--require-full")
    for name, config in [("baseline", "configs/baseline.yaml"), ("residual", "configs/improved.yaml")]:
        destination = output / name
        if not (destination / "frozen.json").exists():
            command = [*cli, "train", "--root", args.data, "--config", config, "--output", destination]
            if (destination / "last.pt").exists():
                command.append("--resume")
            run(*command)
    study = output / "study.json"
    if not study.exists():
        run(*cli, "freeze", "--runs", output / "baseline", output / "residual", "--output", study)
    run(*cli, "evaluate", "--root", args.data, "--study", study, "--output", output / "reports")
    bundle = output / "demo"
    if not (bundle / "manifest.json").exists():
        run(*cli, "bundle", "--root", args.data, "--study", study, "--reports", output / "reports", "--output", bundle)
    print(f"Complete. Reports: {output / 'reports'}; release: {output / 'demo.tar.gz'}", flush=True)


if __name__ == "__main__":
    main()
