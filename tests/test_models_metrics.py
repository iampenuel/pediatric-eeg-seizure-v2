import numpy as np
import pytest
import torch
from seizure_v2.models import make_model
from seizure_v2.evaluation.metrics import calculate, select_threshold, grouped_metrics
from seizure_v2.data.normalization import fit_scaler
from seizure_v2.evaluation.report import representative


@pytest.mark.parametrize("name", ["baseline", "residual"])
def test_model_shapes_and_gradient(name):
    torch.set_num_threads(2)
    model = make_model(name)
    output = model(torch.randn(3, 18, 2048))
    assert output.shape == (3,)
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.sigmoid(output).detach().numpy().min() >= 0


def test_metrics_with_known_confusion_and_undefined_values():
    m = calculate([0, 0, 1, 1], [.1, .8, .4, .9], .5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["f1"] == m["specificity"] == m["sensitivity"] == .5
    single = calculate([0, 0], [.1, .2], .5)
    assert single["auroc"] is None and single["auprc_ap"] is None
    assert single["sensitivity"] is None and single["precision"] is None


def test_threshold_only_validation_and_high_tie():
    rows = [{"partition": "validation", "individual_id": "a", "label": y} for y in [0, 1]]
    selected = select_threshold(rows, [.1, .8])
    assert selected["threshold"] == .8
    rows[0]["partition"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_threshold(rows, [.1, .8])
    with pytest.raises(ValueError, match="training"):
        fit_scaler("unused", rows, "fixture")


def test_macro_counts_chb01_and_21_as_one_person():
    rows = [{"individual_id": "chb01", "case_id": case, "label": y} for case,y in [("chb01",1),("chb21",0)]]
    grouped = grouped_metrics(rows, [.9,.2], .5)
    assert len(grouped) == 1 and grouped[0]["n_windows"] == 2


@pytest.mark.parametrize("layout", ["channels_samples_v1", "samples_channels_v1"])
def test_scaler_uses_only_complete_training_samples(tmp_path, layout):
    from seizure_v2.common import write_json
    from seizure_v2.data.edf import cache_paths
    values = np.tile(np.arange(2048, dtype=np.int16), (18, 1))
    # An incomplete trailing segment must not influence fitted statistics.
    values = np.concatenate([values, np.full((18, 31), 30000, dtype=np.int16)], axis=1)
    array, metadata = cache_paths(tmp_path, "chb02/example.edf")
    array.parent.mkdir(parents=True)
    np.save(array, values if layout == "channels_samples_v1" else values.T.copy())
    write_json(metadata, {"calibration": [{"scale_uv": 2, "offset_uv": 3}] * 18, "cache_layout": layout})
    rows = [{"partition": "train", "recording": "chb02/example.edf", "end_sample": 2048}]
    fitted = fit_scaler(tmp_path, rows, "fixture")
    expected = np.arange(2048) * 2 + 3
    np.testing.assert_allclose(fitted["mean_uv"], expected.mean())
    np.testing.assert_allclose(fitted["std_uv"], expected.std())
    assert fitted["sample_count_per_channel"] == 2048


def test_invalid_resume_preserves_scaler(tmp_path, monkeypatch):
    import importlib
    import yaml
    training = importlib.import_module("seizure_v2.training.train")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model": "baseline"}))
    scaler = tmp_path / "scaler.json"
    scaler.write_text("original frozen scaler")
    monkeypatch.setattr(training, "audit_dataset", lambda *args, **kwargs: {"dataset_hash": "new"})
    monkeypatch.setattr(training.torch, "load", lambda *args, **kwargs: {"dataset_hash": "old", "config": {"model": "baseline"}})
    with pytest.raises(ValueError, match="identical dataset"):
        training.train(tmp_path, config, tmp_path, resume=True, development=True)
    assert scaler.read_text() == "original frozen scaler"


@pytest.mark.parametrize("layout", ["channels_samples_v1", "samples_channels_v1"])
def test_random_io_hint_preserves_shuffled_windows_and_source(tmp_path, layout):
    from seizure_v2.common import write_json, sha256
    from seizure_v2.data.edf import cache_paths
    from seizure_v2.data.channels import CHANNELS
    from seizure_v2.training.dataset import WindowDataset
    rows = []
    for number in range(10):  # Force eviction from the bounded mapping cache.
        recording = f"chb02/fixture{number}.edf"
        array, metadata = cache_paths(tmp_path, recording)
        array.parent.mkdir(parents=True, exist_ok=True)
        values = np.random.default_rng(number).integers(-32768, 32768, (18, 4096), dtype=np.int16)
        np.save(array, values if layout == "channels_samples_v1" else values.T.copy())
        write_json(metadata, {"calibration": [{"scale_uv": .13, "offset_uv": -2.5}] * 18, "cache_layout": layout})
        rows += [{"recording": recording, "start_sample": start, "end_sample": start + 2048,
                  "label": number % 2} for start in [0, 2048]]
    scaler = {"channels": CHANNELS, "fit_partition": "train", "mean_uv": [1.] * 18, "std_uv": [37.] * 18}
    ordinary = WindowDataset(tmp_path, rows, scaler)
    advised = WindowDataset(tmp_path, rows, scaler, random_access=True)
    before = {str(p): sha256(p) for p in tmp_path.rglob("*.npy")}
    for index in np.random.default_rng(42).permutation(len(rows)).tolist() * 2:
        x, y = ordinary[index]
        hinted_x, hinted_y = advised[index]
        np.testing.assert_array_equal(x, hinted_x)
        assert y == hinted_y
    assert before == {str(p): sha256(p) for p in tmp_path.rglob("*.npy")}


def test_representatives_include_failures_with_provenance():
    rows = [{"window_id": f"{i}", "recording": f"r{i}", "individual_id": f"p{i}", "label": y} for i,y in enumerate([1,0,0,1])]
    selected = representative(rows, [.9,.1,.8,.2], .5)
    assert {r["category"] for r in selected} == {"TP","TN","FP","FN"}
    assert all("score" in r and "threshold" in r for r in selected)
