"""Exercise an actual interruption at the same boundary used by Colab backups."""
import importlib
import json
import random

import numpy as np
import pytest
import torch
import yaml


def test_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    training = importlib.import_module("seizure_v2.training.train")
    rows = {
        partition: [
            {"partition": partition, "individual_id": person, "label": index % 2}
            for index in range(4)
        ]
        for partition, person in [("train", "chb02"), ("validation", "chb05")]
    }

    class SyntheticWindows(torch.utils.data.Dataset):
        def __init__(self, root, selected, scaler, **kwargs):
            self.rows = selected

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            # Exercise all saved CPU RNG streams, plus Dropout in the real model.
            signal = torch.randn(18, 2048) * .01
            signal += random.random() * .01 + np.random.random() * .01
            return signal, np.float32(self.rows[index]["label"])

    monkeypatch.setattr(training, "audit_dataset", lambda *a, **k: {
        "dataset_hash": "resume-fixture", "split_sha256": "split-fixture"})
    monkeypatch.setattr(training, "load_rows", lambda root, partition: rows[partition])
    monkeypatch.setattr(training, "fit_scaler", lambda *a: {"fit_partition": "train"})
    monkeypatch.setattr(training, "WindowDataset", SyntheticWindows)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "model": "baseline", "seed": 42, "learning_rate": .001,
        "batch_size": 2, "max_epochs": 2, "patience": 4,
        "num_workers": 0, "threshold_step": .001,
    }))
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    monkeypatch.setattr(training, "checkpoint", lambda stage: None)
    training.train(tmp_path, config, complete, development=True, device="cpu")

    class Interruption(Exception):
        pass

    def stop_after_epoch(stage):
        if stage == "baseline-epoch-01":
            raise Interruption

    monkeypatch.setattr(training, "checkpoint", stop_after_epoch)
    with pytest.raises(Interruption):
        training.train(tmp_path, config, interrupted, development=True, device="cpu")
    assert (interrupted / "last.pt").exists()
    assert not (interrupted / "frozen.json").exists()
    monkeypatch.setattr(training, "checkpoint", lambda stage: None)
    training.train(tmp_path, config, interrupted, resume=True, development=True, device="cpu")

    reference = torch.load(complete / "last.pt", weights_only=False)
    resumed = torch.load(interrupted / "last.pt", weights_only=False)
    assert resumed["epoch"] == reference["epoch"] == 1
    assert resumed["best"] == reference["best"]
    assert resumed["stale"] == reference["stale"]
    for name in reference["model_state"]:
        assert torch.equal(reference["model_state"][name], resumed["model_state"][name])
    for key in ["torch_rng", "loader_rng"]:
        assert torch.equal(reference[key], resumed[key])
    assert reference["python_rng"] == resumed["python_rng"]
    assert np.array_equal(reference["numpy_rng"][1], resumed["numpy_rng"][1])
    for key, state in reference["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed["optimizer"]["state"][key][name])
    for actual, expected in zip(resumed["history"], reference["history"], strict=True):
        assert {k: v for k, v in actual.items() if k != "seconds"} == {
            k: v for k, v in expected.items() if k != "seconds"}
    frozen = [json.loads((p / "frozen.json").read_text()) for p in [complete, interrupted]]
    for key in ["threshold", "checkpoint_epoch", "validation_patient_macro_ap", "validation_patient_macro_f1"]:
        assert frozen[0][key] == frozen[1][key]
    run = json.loads((interrupted / "run.json").read_text())
    assert [segment["first_epoch"] for segment in run["execution_segments"]] == [1, 2]
