import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from seizure_v2.common import read_json, write_json, sha256, object_hash
from seizure_v2.data.prepare import audit_dataset, load_rows
from seizure_v2.data.edf import read_window
from seizure_v2.data.channels import CHANNELS
from seizure_v2.models import make_model
from seizure_v2.training.dataset import WindowDataset
from seizure_v2.training.train import predict, choose_device
from seizure_v2.evaluation.metrics import calculate, grouped_metrics, macro_average


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_study(path):
    study = read_json(path)
    digest = study.pop("sha256")
    if object_hash(study) != digest:
        raise ValueError("Frozen study was modified")
    study["sha256"] = digest
    for model in study["models"].values():
        run = Path(path).parent / model["run_dir"]
        if sha256(run / "best.pt") != model["checkpoint_sha256"] or sha256(run / "scaler.json") != model["scaler_sha256"]:
            raise ValueError("Frozen checkpoint or scaler was modified")
    return study


def representative(rows, scores, threshold, limit_per_category=4):
    grouped = {category: [] for category in ["TP", "TN", "FP", "FN"]}
    for row, score in zip(rows, scores, strict=True):
        predicted = int(score >= threshold)
        category = ("T" if predicted == row["label"] else "F") + ("P" if predicted else "N")
        grouped[category].append({**row, "score": float(score), "predicted_label": predicted,
                                  "threshold": threshold, "category": category})
    result = []
    for category, candidates in grouped.items():
        candidates.sort(key=lambda row: ((-row["score"] if category.endswith("P") else row["score"]), row["window_id"]))
        used_people, used_recordings, chosen = set(), set(), []
        for unique_people in [True, False]:
            for row in candidates:
                if len(chosen) >= limit_per_category:
                    break
                if row["recording"] in used_recordings or (unique_people and row["individual_id"] in used_people):
                    continue
                chosen.append(row)
                used_people.add(row["individual_id"])
                used_recordings.add(row["recording"])
        result.extend(chosen)
    return result


def waveform_figure(signal, start_seconds, title, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(12, 8))
    time = np.arange(signal.shape[1]) / 256 + start_seconds
    spacing = max(float(np.percentile(np.abs(signal - np.median(signal, axis=1, keepdims=True)), 98)) * 2.5, 20)
    for i, channel in enumerate(CHANNELS):
        axis.plot(time, signal[i] - np.median(signal[i]) + (17 - i) * spacing, linewidth=.45, color="#163f47")
    axis.set_yticks(np.arange(18) * spacing, CHANNELS[::-1])
    axis.set_xlabel("Seconds from recording start")
    axis.set_title(title + f"\nDisplay offsets {spacing:.0f} µV; per-channel median removed for display only", fontsize=10)
    axis.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def figures(rows, scores, threshold, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    y = np.array([row["label"] for row in rows])
    fig, axis = plt.subplots(figsize=(4, 4))
    matrix = np.asarray(metrics["confusion_matrix"])
    axis.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i,j] > matrix.max()/2 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-seizure", "Seizure"], yticklabels=["Non-seizure", "Seizure"], xlabel="Predicted", ylabel="Annotated", title=f"Threshold {threshold:.3f}")
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=140); plt.close(fig)
    if len(set(y)) == 2:
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        write_csv(output / "roc.csv", [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)])
        write_csv(output / "precision_recall.csv", [{"recall": float(a), "precision": float(b)} for a, b in zip(recall, precision)])
        for name, x, v, xlabel, ylabel in [("roc", fpr, tpr, "False positive rate", "Sensitivity"), ("precision_recall", recall, precision, "Recall", "Precision")]:
            fig, axis = plt.subplots(figsize=(5, 4))
            axis.plot(x, v, color="#146c68")
            if name == "precision_recall":
                axis.axhline(y.mean(), color="#bb793b", linestyle="--", label="Window prevalence")
                axis.legend()
            axis.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
            fig.tight_layout(); fig.savefig(output / f"{name}.png", dpi=140); plt.close(fig)


def evaluate(root, study, output, validation_only=False, device="auto"):
    path = Path(study)
    spec = verify_study(path)
    if spec["development"] and not validation_only:
        raise ValueError("Development runs cannot produce final held-out test reports")
    dataset = audit_dataset(root, require_full=not spec["development"])
    output = Path(output)
    device = choose_device(device)
    torch.set_num_threads(4)
    all_reports = {}
    for name, frozen in spec["models"].items():
        if frozen["dataset_hash"] != dataset["dataset_hash"]:
            raise ValueError("Evaluation data differs from frozen experiment")
        run = path.parent / frozen["run_dir"]
        checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
        model = make_model(name).to(device)
        model.load_state_dict(checkpoint["model_state"])
        scaler = read_json(run / "scaler.json")
        for partition in (["validation"] if validation_only else ["validation", "test"]):
            rows = load_rows(root, partition)
            loader = DataLoader(WindowDataset(root, rows, scaler), batch_size=128, shuffle=False)
            scores = predict(model, loader, device)
            threshold = frozen["threshold"]
            patients = grouped_metrics(rows, scores, threshold)
            report = {"model": name, "partition": partition, "development": spec["development"],
                      "study_sha256": spec["sha256"], "dataset_hash": dataset["dataset_hash"],
                      "pooled": calculate([row["label"] for row in rows], scores, threshold),
                      "patient_macro": macro_average(patients), "patients": patients,
                      "cases": grouped_metrics(rows, scores, threshold, "case_id"),
                      "metric_note": "AUPRC/AP uses average precision; undefined metrics are null. These are window metrics, not event sensitivity or false alarms/hour."}
            destination = output / name / partition
            write_json(destination / "metrics.json", report)
            write_csv(destination / "patients.csv", [{k:v for k,v in p.items() if k != "confusion_matrix"} for p in patients])
            write_csv(destination / "cases.csv", [{k:v for k,v in p.items() if k != "confusion_matrix"} for p in report["cases"]])
            predictions = [{**row, "score": float(score), "predicted_label": int(score >= threshold), "threshold": threshold} for row, score in zip(rows, scores, strict=True)]
            write_csv(destination / "predictions.csv", predictions)
            figures(rows, scores, threshold, report["pooled"], destination / "figures")
            selected = representative(rows, scores, threshold)
            write_json(destination / "representative_examples.json", selected)
            failures = [row for row in selected if row["category"] in {"FP", "FN"}]
            write_csv(destination / "failures.csv", failures)
            (destination / "failures").mkdir(exist_ok=True)
            for i, row in enumerate(failures):
                signal = read_window(root, row["recording"], row["start_sample"], row["end_sample"])
                waveform_figure(signal, row["start_seconds"], f"{row['category']} · {row['recording']} · score {row['score']:.3f} · true {row['label']}", destination / "failures" / f"{i:02d}-{row['category']}.png")
            write_json(destination / "failure_notes.json", {"observations": "See exported waveforms and scored provenance. Error category describes the model-versus-annotation disagreement only.", "hypotheses": [], "note": "No clinical interpretation or causal explanation has been inferred automatically."})
            all_reports[f"{name}/{partition}"] = report
    write_json(output / "summary.json", all_reports)
    return {"reports": sorted(all_reports), "output": str(output), "development": spec["development"]}
