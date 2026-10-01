import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Patient-independent CHB-MIT research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", default="data")
    choice = prep.add_mutually_exclusive_group(required=True)
    choice.add_argument("--patients", nargs="+")
    choice.add_argument("--recordings", nargs="+")
    choice.add_argument("--full", action="store_true")
    prep.add_argument("--evict-raw", action="store_true")
    prep.add_argument("--mirror", choices=["physionet", "s3"], default="physionet")
    prep.add_argument("--workers", type=int, default=1)
    audit = sub.add_parser("audit")
    audit.add_argument("--root", default="data")
    audit.add_argument("--require-full", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--root", default="data")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--development", action="store_true")
    train.add_argument("--epochs", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--device", default="auto")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--runs", nargs=2, required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--development", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--root", default="data")
    evaluate.add_argument("--study", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--validation-only", action="store_true")
    evaluate.add_argument("--device", default="auto")
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--root", default="data")
    bundle.add_argument("--study", required=True)
    bundle.add_argument("--reports", required=True)
    bundle.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from seizure_v2.data.prepare import prepare
        result = prepare(**args)
    elif command == "audit":
        from seizure_v2.data.prepare import audit_dataset
        result = audit_dataset(**args)
    elif command == "train":
        from seizure_v2.training.train import train
        result = train(**args)
    elif command == "freeze":
        from seizure_v2.training.train import freeze_study
        result = freeze_study(**args)
    elif command == "evaluate":
        from seizure_v2.evaluation.report import evaluate
        result = evaluate(**args)
    elif command == "bundle":
        from seizure_v2.inference.bundle import build_bundle
        result = build_bundle(**args)
    print(json.dumps({k: v for k, v in result.items() if k not in {"prepared", "exclusions"}}, indent=2))


if __name__ == "__main__":
    main()
