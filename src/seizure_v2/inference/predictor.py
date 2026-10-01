from pathlib import Path
import numpy as np
from seizure_v2.common import read_json, sha256
from seizure_v2.data.normalization import normalize


class Predictor:
    def __init__(self, bundle):
        import onnxruntime as ort
        self.root = Path(bundle)
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("development", True):
            raise ValueError("Development artifacts cannot back the public research demo")
        for name, digest in self.manifest["files"].items():
            candidate = (self.root / name).resolve()
            if not candidate.is_relative_to(self.root.resolve()) or sha256(candidate) != digest:
                raise ValueError(f"Artifact failed verification: {name}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.sessions = {name: ort.InferenceSession(str(self.root / f"models/{name}.onnx"), sess_options=options, providers=["CPUExecutionProvider"]) for name in self.manifest["models"]}
        self.scalers = {name: read_json(self.root / f"models/{name}-scaler.json") for name in self.sessions}
        self.examples = {example["id"]: example for example in self.manifest["examples"]}

    def predict(self, model_id, signal_uv):
        if model_id not in self.sessions:
            raise KeyError("Unknown model")
        x = normalize(signal_uv, self.scalers[model_id])
        chunks = []
        for start in range(0, len(x), 16):
            logits = self.sessions[model_id].run(None, {"eeg": x[start:start + 16]})[0]
            chunks.extend((1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))).tolist())
        scores = np.asarray(chunks)
        threshold = self.manifest["models"][model_id]["threshold"]
        return scores, (scores >= threshold).astype(int), threshold

    def clip(self, example_id):
        example = self.examples[example_id]
        with np.load(self.root / example["file"], allow_pickle=False) as value:
            return example, value["signal_uv"].copy(), value["labels"].copy()

    def predict_example(self, example_id, model_id):
        example, signal, labels = self.clip(example_id)
        windows = signal.reshape(18, -1, 2048).transpose(1, 0, 2)
        scores, predicted, threshold = self.predict(model_id, windows)
        output = [{"start_seconds": example["clip_start_seconds"] + 8 * i,
                   "end_seconds": example["clip_start_seconds"] + 8 * (i + 1),
                   "score": float(score), "predicted_label": int(predicted[i]), "true_label": int(labels[i])}
                  for i, score in enumerate(scores)]
        regions = []
        for window in output:
            if window["predicted_label"]:
                if regions and regions[-1][1] == window["start_seconds"]:
                    regions[-1][1] = window["end_seconds"]
                else:
                    regions.append([window["start_seconds"], window["end_seconds"]])
        return {"model_id": model_id, "example_id": example_id, "threshold": threshold,
                "windows": output, "high_score_regions": regions, "score_type": "uncalibrated_seizure_score"}
