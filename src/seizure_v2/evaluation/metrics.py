import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auroc", "auprc_ap"]


def validate_scores(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if y.shape != score.shape or y.ndim != 1 or not len(y):
        raise ValueError("Nonempty one-dimensional labels and scores required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Invalid labels or scores")
    return y, score


def calculate(y, score, threshold):
    y, score = validate_scores(y, score)
    predicted = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist()
    ratio = lambda a, b: a / b if b else None
    both = len(set(y)) == 2
    return {"n_windows": len(y), "positive_windows": int(y.sum()), "negative_windows": int((1-y).sum()),
            "duration_hours": len(y) * 8 / 3600, "prevalence": float(y.mean()),
            "accuracy": ratio(tp + tn, len(y)), "sensitivity": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp), "precision": ratio(tp, tp + fp),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "auroc": float(roc_auc_score(y, score)) if both else None,
            "auprc_ap": float(average_precision_score(y, score)) if both else None,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "confusion_matrix": [[tn, fp], [fn, tp]],
            "threshold": float(threshold)}


def grouped_metrics(rows, scores, threshold, group_key="individual_id"):
    groups = sorted({row[group_key] for row in rows})
    result = []
    scores = np.asarray(scores)
    for group in groups:
        indices = [i for i, row in enumerate(rows) if row[group_key] == group]
        result.append({group_key: group, **calculate([rows[i]["label"] for i in indices], scores[indices], threshold)})
    return result


def macro_average(groups):
    return {key: float(np.mean(values)) if (values := [g[key] for g in groups if g[key] is not None]) else None for key in METRIC_KEYS}


def select_threshold(rows, scores, step=.001):
    if not rows or any(row["partition"] != "validation" for row in rows):
        raise ValueError("Threshold selection accepts validation rows only")
    y, scores = validate_scores([row["label"] for row in rows], scores)
    thresholds = np.linspace(0, 1, round(1 / step) + 1)
    macro_f1 = np.zeros(len(thresholds))
    groups = sorted({row["individual_id"] for row in rows})
    for group in groups:
        mask = np.array([row["individual_id"] == group for row in rows])
        labels, values = y[mask], scores[mask]
        if labels.sum() == 0:
            raise ValueError("Each validation individual must have positive windows")
        order = np.argsort(values, kind="stable")
        labels, values = labels[order], values[order]
        prefix = np.r_[0, np.cumsum(labels)]
        cut = np.searchsorted(values, thresholds, side="left")
        tp = prefix[-1] - prefix[cut]
        fp = len(labels) - cut - tp
        fn = prefix[cut]
        macro_f1 += 2 * tp / (2 * tp + fp + fn) / len(groups)
    # Fixed grid; highest threshold wins exact numerical ties.
    best = np.flatnonzero(np.isclose(macro_f1, macro_f1.max(), rtol=0, atol=1e-12))[-1]
    return {"threshold": float(thresholds[best]), "validation_patient_macro_f1": float(macro_f1[best]),
            "selection_partition": "validation", "rule": "patient_macro_f1", "grid_step": step, "tie_break": "higher_threshold"}
