import shutil
import tarfile
from pathlib import Path
import numpy as np
import torch
from seizure_v2.common import read_json, write_json, sha256
from seizure_v2.data.edf import read_window, cache_paths
from seizure_v2.data.normalization import normalize
from seizure_v2.data.windows import window_rows
from seizure_v2.evaluation.report import verify_study
from seizure_v2.models import make_model


def export_onnx(model, path, reference_inputs):
    import onnxruntime as ort
    model = model.cpu().eval()
    torch.onnx.export(model, torch.from_numpy(reference_inputs[:1]), str(path),
                      input_names=["eeg"], output_names=["logits"],
                      dynamic_axes={"eeg": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = torch.sigmoid(model(torch.from_numpy(reference_inputs))).numpy()
    logits = runtime.run(None, {"eeg": reference_inputs})[0]
    actual = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
    error = float(np.max(np.abs(expected - actual)))
    if error > 1e-5:
        raise ValueError(f"ONNX score parity failed: {error}")
    return {"maximum_score_error": error, "tolerance": 1e-5, "reference_windows": len(reference_inputs)}


def build_bundle(root, study, reports, output):
    spec = verify_study(study)
    if spec["development"]:
        raise ValueError("Only a final full-cohort study can be published")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Bundle destination must be empty")
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "examples").mkdir(exist_ok=True)
    examples, selected_ids, models = [], set(), {}
    for name, frozen in spec["models"].items():
        test_dir = Path(reports) / name / "test"
        metrics = read_json(test_dir / "metrics.json")
        if metrics["study_sha256"] != spec["sha256"] or metrics["development"] or metrics["partition"] != "test":
            raise ValueError("Reports do not match the frozen final study")
        run = Path(study).parent / frozen["run_dir"]
        scaler = read_json(run / "scaler.json")
        model = make_model(name)
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model_state"])
        # Export checks use validation windows only.
        from seizure_v2.data.prepare import load_rows
        validation = load_rows(root, "validation")
        indices = np.linspace(0, len(validation) - 1, min(32, len(validation)), dtype=int)
        x = np.stack([normalize(read_window(root, validation[i]["recording"], validation[i]["start_sample"], validation[i]["end_sample"]), scaler) for i in indices])
        parity = export_onnx(model, output / f"models/{name}.onnx", x)
        shutil.copyfile(run / "scaler.json", output / f"models/{name}-scaler.json")
        models[name] = {"threshold": frozen["threshold"], "metrics": metrics,
                        "validation_patient_macro_ap": frozen["validation_patient_macro_ap"], "export_parity": parity}
        for chosen in read_json(test_dir / "representative_examples.json"):
            key = chosen["window_id"]
            if key in selected_ids:
                continue
            selected_ids.add(key)
            meta = read_json(cache_paths(root, chosen["recording"])[1])
            start = max(0, chosen["start_sample"] - 7 * 2048)
            end = min(meta["n_samples"] // 2048 * 2048, start + 16 * 2048)
            signal = read_window(root, chosen["recording"], start, end)
            labels = [row["label"] for row in window_rows(meta) if start <= row["start_sample"] < end]
            identifier = f"example-{len(examples) + 1:03d}"
            file = f"examples/{identifier}.npz"
            np.savez_compressed(output / file, signal_uv=signal, labels=np.asarray(labels, dtype=np.int8))
            examples.append({"id": identifier, "file": file, "individual_id": chosen["individual_id"],
                             "case_id": chosen["case_id"], "recording": chosen["recording"], "partition": "test",
                             "clip_start_seconds": start / 256, "clip_end_seconds": end / 256,
                             "focus_start_seconds": chosen["start_seconds"], "seizures": meta["seizures"],
                             "curation_model": name, "curation_category": chosen["category"],
                             "source_sha256": meta["source_sha256"], "source_url": meta["source_url"]})
    write_json(output / "study.json", spec)
    attribution = "Derived clips from CHB-MIT Scalp EEG Database v1.0.0, John Guttag (2010), PhysioNet. DOI: 10.13026/C2K01R. Open Data Commons Attribution License v1.0. https://physionet.org/content/chbmit/1.0.0/ . Clips are curated illustrations, not a representative evaluation sample."
    (output / "ATTRIBUTION.txt").write_text(attribution + "\n")
    manifest = {"development": False, "version": 1, "study_sha256": spec["sha256"],
                "default_model": spec["default_model"], "models": models, "examples": examples,
                "attribution": attribution,
                "files": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for item in sorted(output.iterdir()):
            tar.add(item, arcname=item.name)
    return {"bundle": str(output), "archive": str(archive), "sha256": sha256(archive), "examples": len(examples)}
