from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def baseline_classification_metrics(labels: Iterable[int], probs: Iterable[float], preds: Iterable[int]) -> Dict[str, float]:
    labels = np.array(list(labels)).astype(int)
    probs = np.array(list(probs)).astype(float)
    preds = np.array(list(preds)).astype(int)
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(labels, preds)) if len(labels) else 0.0
    metrics["precision"] = float(precision_score(labels, preds, zero_division=0)) if len(labels) else 0.0
    metrics["recall"] = float(recall_score(labels, preds, zero_division=0)) if len(labels) else 0.0
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0)) if len(labels) else 0.0
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    except Exception:
        metrics["roc_auc"] = 0.0
    if len(labels):
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics["specificity"] = float(tn / max(1, tn + fp))
        metrics["sensitivity"] = float(tp / max(1, tp + fn))
    else:
        metrics["specificity"] = 0.0
        metrics["sensitivity"] = 0.0
    return metrics


def prediction_flip_rate(original_preds: Iterable[int], perturbed_preds: Iterable[int]) -> float:
    a = np.array(list(original_preds)).astype(int)
    b = np.array(list(perturbed_preds)).astype(int)
    if len(a) == 0:
        return 0.0
    return float(np.mean(a != b))


def mean_confidence_drop(original_probs: Iterable[float], perturbed_probs: Iterable[float]) -> float:
    a = np.array(list(original_probs)).astype(float)
    b = np.array(list(perturbed_probs)).astype(float)
    if len(a) == 0:
        return 0.0
    return float(np.mean(b - a))


def explanation_similarity(map_a: np.ndarray, map_b: np.ndarray) -> float:
    a = map_a.astype(np.float32)
    b = map_b.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b = (b - b.min()) / (b.max() - b.min() + 1e-8)
    return float(np.mean(1.0 - np.abs(a - b)))


def explanation_shift(map_a: np.ndarray, map_b: np.ndarray) -> float:
    return float(1.0 - explanation_similarity(map_a, map_b))


def summarize_audit_rows(rows: List[Dict]) -> Dict[str, float]:
    if len(rows) == 0:
        return {"prediction_flip_rate": 0.0, "mean_confidence_drop": 0.0, "mean_explanation_shift": 0.0}
    original_preds = [r["original_pred"] for r in rows]
    perturbed_preds = [r["perturbed_pred"] for r in rows]
    original_probs = [r["original_prob_tumor"] for r in rows]
    perturbed_probs = [r["perturbed_prob_tumor"] for r in rows]
    explanation_shifts = [r["explanation_shift"] for r in rows]
    return {
        "prediction_flip_rate": prediction_flip_rate(original_preds, perturbed_preds),
        "mean_confidence_drop": mean_confidence_drop(original_probs, perturbed_probs),
        "mean_explanation_shift": float(np.mean(explanation_shifts)),
    }
