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


def test_representatives_include_failures_with_provenance():
    rows = [{"window_id": f"{i}", "recording": f"r{i}", "individual_id": f"p{i}", "label": y} for i,y in enumerate([1,0,0,1])]
    selected = representative(rows, [.9,.1,.8,.2], .5)
    assert {r["category"] for r in selected} == {"TP","TN","FP","FN"}
    assert all("score" in r and "threshold" in r for r in selected)
