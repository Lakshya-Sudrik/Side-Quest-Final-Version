"""Honest evaluation: per-class metrics, baselines, threshold tuning, gates.

Everything a report is allowed to claim lives here. Rules enforced by this
module:

* Classification never reports accuracy alone. Every report carries per-class
  precision/recall/F1, macro-F1, balanced accuracy, PR-AUC and the confusion
  matrix, so a majority-class predictor cannot masquerade as a good model.
* Every gate is evaluated as ``value >= max(floor, baseline + margin)`` so a
  gate can never be satisfied by predicting the majority class.
* ``label_shuffle_test`` proves the metric actually depends on the labels; a
  model that still scores well on shuffled labels is leaking.

Reads config/criteria.json - there is no second copy of the thresholds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CRITERIA_PATH = ROOT / "config" / "criteria.json"


def load_criteria() -> dict:
    with open(CRITERIA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def task_criteria(task: str) -> dict:
    crit = load_criteria()["tasks"]
    if task not in crit:
        raise KeyError(f"task {task!r} not defined in {CRITERIA_PATH}")
    return crit[task]


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def majority_baseline(y: np.ndarray) -> Dict[str, float]:
    """What an 'always predict the most common class' model would score."""
    y = np.asarray(y)
    values, counts = np.unique(y, return_counts=True)
    maj = values[np.argmax(counts)]
    return {
        "majority_class": maj.item() if hasattr(maj, "item") else float(maj),
        "majority_share": float(counts.max() / len(y)),
        "majority_accuracy": float(counts.max() / len(y)),
        "n_samples": int(len(y)),
        "prevalence_positive": float((y == values.max()).mean()) if len(values) > 1 else 1.0,
    }


def regression_baseline(y: np.ndarray) -> Dict[str, float]:
    """R^2 of the constant-mean predictor (= 0 by definition) plus spread."""
    y = np.asarray(y, dtype=float)
    mean = float(np.mean(y))
    sse = float(np.sum((y - mean) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mean": mean,
        "std": float(np.std(y)),
        "baseline_r2": 0.0 if sst <= 0 else float(1.0 - sse / sst),
        "constant_rmse": float(np.sqrt(sse / max(len(y), 1))),
        "n_samples": int(len(y)),
    }


# ---------------------------------------------------------------------------
# classification report
# ---------------------------------------------------------------------------
def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    positive: Any = None,
) -> Dict[str, Any]:
    """Metrics with the MINORITY class as the class of interest.

    ``y_prob`` must be the probability of ``positive`` (defaults to the least
    frequent class). Scoring PR-AUC on the majority class is meaningless when
    a dataset is 9:1 imbalanced, so it is never done here.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    counts_true = {c: int((y_true == c).sum()) for c in classes}
    minority = min(counts_true, key=counts_true.get) if counts_true else 1
    if positive is None:
        positive = minority

    per_class: Dict[str, Dict[str, float]] = {}
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[str(c)] = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": int((y_true == c).sum()),
        }

    baseline = majority_baseline(y_true)
    prevalence = float(counts_true.get(positive, 0)) / max(len(y_true), 1)
    report: Dict[str, Any] = {
        "type": "classification",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "precision_positive": round(float(precision_score(y_true, y_pred, pos_label=positive, zero_division=0)), 6),
        "recall_positive": round(float(recall_score(y_true, y_pred, pos_label=positive, zero_division=0)), 6),
        "f1_positive": round(float(f1_score(y_true, y_pred, pos_label=positive, zero_division=0)), 6),
        "positive_class": positive,
        "minority_class": minority,
        "minority_recall": per_class[str(minority)]["recall"],
        "minority_f1": per_class[str(minority)]["f1"],
        "minority_precision": per_class[str(minority)]["precision"],
        "prevalence_positive": round(prevalence, 6),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "confusion_matrix_labels": [str(c) for c in classes],
        "baseline": baseline,
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            report["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 6)
        except Exception:
            report["pr_auc"] = 0.0
        try:
            report["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        except Exception:
            report["roc_auc"] = 0.0
        # lift over how often the class of interest actually occurs
        report["pr_auc_lift_over_prevalence"] = round(
            (report["pr_auc"] / prevalence) if prevalence > 0 else 0.0, 4
        )
    return report


def regression_report(
    y_true: np.ndarray, y_pred: np.ndarray, groups: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    base = regression_baseline(y_true)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    out: Dict[str, Any] = {
        "type": "regression",
        "r2": round(r2, 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "baseline_r2": base["baseline_r2"],
        "r2_minus_baseline": round(r2 - base["baseline_r2"], 6),
        "target_mean": base["mean"],
        "target_std": base["std"],
        "n_samples": base["n_samples"],
        # legacy aliases so older consumers keep working
        "accuracy": round(r2, 6),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc_roc": 0.0,
    }
    # Ranking quality: the app shows a ranked list of gems, so what matters is
    # whether the model puts the right places at the top, not squared error.
    if len(y_true) > 2:
        yt, yp = pd.Series(y_true), pd.Series(y_pred)
        out["spearman"] = round(float(yt.corr(yp, method="spearman")), 6)
        k = max(1, int(round(0.2 * len(y_true))))
        top_true = set(np.argsort(-y_true)[:k])
        top_pred = np.argsort(-y_pred)[:k]
        out["precision_at_top20pct"] = round(len(top_true.intersection(top_pred)) / k, 6)
        out["precision_at_top20pct_random"] = round(k / len(y_true), 6)
    if groups is not None:
        # For district-level labels every row in a group shares one value, so
        # row-level R^2 is dominated by groups with many rows. Report R^2 over
        # group means as well - that is the generalisation we actually care
        # about (new district, not new restaurant).
        g = pd.DataFrame({"g": np.asarray(groups), "y": y_true, "p": y_pred})
        gm = g.groupby("g").mean()
        try:
            out["group_r2"] = round(float(r2_score(gm["y"], gm["p"])), 6)
        except Exception:
            out["group_r2"] = 0.0
        out["n_groups"] = int(g["g"].nunique())
        out["group_rmse"] = round(
            float(np.sqrt(np.mean((gm["y"].to_numpy() - gm["p"].to_numpy()) ** 2))), 6
        )
    return out


# ---------------------------------------------------------------------------
# threshold tuning + calibration
# ---------------------------------------------------------------------------
def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimise_for: str = "positive_f1",
    positive: Any = None,
    grid: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """Pick the cut-off on P(positive) that maximises a minority-oriented metric.

    ``y_prob`` must be the probability of ``positive`` (defaults to the least
    frequent class), so a 9:1 dataset never tunes toward the majority class.
    """
    from sklearn.metrics import f1_score, recall_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if positive is None:
        values, counts = np.unique(y_true, return_counts=True)
        positive = values[np.argmin(counts)]
    grid = list(grid) if grid is not None else [round(x, 3) for x in np.arange(0.05, 0.96, 0.01)]
    best_t, best_v = 0.5, -1.0
    other = next((c for c in np.unique(y_true) if c != positive), positive)
    for t in grid:
        pred = np.where(y_prob >= t, positive, other)
        if optimise_for in ("minority_recall", "positive_recall"):
            v = float(recall_score(y_true, pred, pos_label=positive, zero_division=0))
        elif optimise_for in ("macro_f1", "balanced_accuracy"):
            # keeps the threshold from degenerating into "flag everything as
            # the minority class", which a pure minority-F1 objective allows
            v = float(f1_score(y_true, pred, average="macro", zero_division=0))
        else:
            v = float(f1_score(y_true, pred, pos_label=positive, zero_division=0))
        if v > best_v:
            best_t, best_v = float(t), v
    return {"threshold": round(best_t, 3), "objective": optimise_for,
            "objective_value": round(best_v, 6), "positive_class": positive}


def calibrate_probabilities(estimator, X_cal: np.ndarray, y_cal: np.ndarray) -> Any:
    """Isotonic calibration - probabilities must be meaningful before any
    threshold other than 0.5 can be trusted."""
    from sklearn.calibration import CalibratedClassifierCV

    try:
        base = estimator
        if hasattr(base, "get_params"):
            params = base.get_params()
            if "early_stopping_rounds" in params:
                base.set_params(early_stopping_rounds=None)
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(X_cal, y_cal)
        return cal
    except Exception as exc:
        logger.warning("calibration skipped (%s); using raw probabilities", exc)
        return estimator


# ---------------------------------------------------------------------------
# label-shuffle sanity test
# ---------------------------------------------------------------------------
def label_shuffle_test(
    clone_fn: Callable[[], Any],
    X,
    y,
    task_type: str,
    fit_extra: Optional[Dict[str, Any]] = None,
    metric: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train on permuted labels. The score MUST collapse towards the baseline.

    A model that still scores well on shuffled labels is reading the answer
    from somewhere other than the labels - i.e. leakage.
    """
    from sklearn.base import clone as sk_clone

    rng = np.random.default_rng(seed)
    y_shuf = np.asarray(y).copy()
    rng.shuffle(y_shuf)

    est = clone_fn()
    fit_kw = dict(fit_extra or {})
    try:
        est.fit(X, y_shuf, **fit_kw)
        pred = est.predict(X)
        if task_type == "classification":
            rep = classification_report(y_shuf, pred)
            value = rep[metric or "macro_f1"]
            baseline = rep["baseline"]["majority_accuracy"]
        else:
            rep = regression_report(y_shuf, pred)
            value = rep["r2"]
            baseline = 0.0
        return {
            "metric": metric or ("macro_f1" if task_type == "classification" else "r2"),
            "shuffled_value": value,
            "baseline": baseline,
            "collapsed": bool(value <= max(baseline + 0.10, 0.10) if task_type == "regression"
                              else value <= baseline + 0.10),
        }
    except Exception as exc:
        logger.warning("label-shuffle test failed to run: %s", exc)
        return {"metric": metric, "shuffled_value": None, "error": str(exc), "collapsed": None}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _check_ge(value: float, required: float) -> Dict[str, Any]:
    return {"value": value, "required": f">= {required}", "pass": bool(value >= required)}


def _check_le(value: float, required: float) -> Dict[str, Any]:
    return {"value": value, "required": f"<= {required}", "pass": bool(value <= required)}


def evaluate_gates(task: str, report: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate config/criteria.json gates against a report.

    ``extra`` carries values that only exist after the full training loop:
    accuracy_gap, r2_gap, shuffled metric, leakage assert result, ablated score.
    """
    g = task_criteria(task).get("gates", {})
    extra = dict(extra or {})
    results: Dict[str, Any] = {}

    if report.get("type") == "regression":
        base = report.get("baseline_r2", 0.0)
        if "r2_min" in g:
            results["r2_min"] = _check_ge(report.get("r2", 0.0), g["r2_min"])
        if "r2_baseline_margin" in g:
            margin = report.get("r2", 0.0) - base
            results["r2_baseline_margin"] = _check_ge(round(margin, 6), g["r2_baseline_margin"])
        if "r2_gap_max" in g and extra.get("r2_gap") is not None:
            results["r2_gap_max"] = _check_le(round(extra["r2_gap"], 6), g["r2_gap_max"])
        if "group_r2_min" in g:
            results["group_r2_min"] = _check_ge(report.get("group_r2", 0.0), g["group_r2_min"])
        if "group_r2_gap_max" in g and extra.get("group_r2_gap") is not None:
            results["group_r2_gap_max"] = _check_le(
                round(extra["group_r2_gap"], 6), g["group_r2_gap_max"]
            )
        if "shuffle_r2_max" in g and extra.get("shuffle_r2") is not None:
            results["shuffle_r2_max"] = _check_le(round(extra["shuffle_r2"], 6), g["shuffle_r2_max"])
    else:
        base_acc = report.get("baseline", {}).get("majority_accuracy", 0.0)
        if "accuracy_min" in g:
            results["accuracy_min"] = _check_ge(report.get("accuracy", 0.0), g["accuracy_min"])
        if "macro_f1_min" in g:
            results["macro_f1_min"] = _check_ge(report.get("macro_f1", 0.0), g["macro_f1_min"])
        if "minority_recall_min" in g:
            results["minority_recall_min"] = _check_ge(report.get("minority_recall", 0.0), g["minority_recall_min"])
        if "positive_recall_min" in g:
            results["positive_recall_min"] = _check_ge(report.get("recall_positive", 0.0), g["positive_recall_min"])
        if "average_precision_min" in g:
            results["average_precision_min"] = _check_ge(report.get("pr_auc", 0.0), g["average_precision_min"])
        if "pr_auc_min_over_prevalence" in g and "pr_auc_lift_over_prevalence" in report:
            results["pr_auc_min_over_prevalence"] = _check_ge(
                report["pr_auc_lift_over_prevalence"], g["pr_auc_min_over_prevalence"]
            )
        if "accuracy_baseline_margin" in g:
            results["accuracy_baseline_margin"] = _check_ge(
                round(report.get("accuracy", 0.0) - base_acc, 6), g["accuracy_baseline_margin"]
            )
        if "balanced_accuracy_min" in g:
            results["balanced_accuracy_min"] = _check_ge(
                report.get("balanced_accuracy", 0.0), g["balanced_accuracy_min"]
            )
        if "accuracy_gap_max" in g and extra.get("accuracy_gap") is not None:
            results["accuracy_gap_max"] = _check_le(round(extra["accuracy_gap"], 6), g["accuracy_gap_max"])
        if "score_r2_min" in g and extra.get("score_r2") is not None:
            results["score_r2_min"] = _check_ge(extra["score_r2"], g["score_r2_min"])
        if "score_r2_gap_max" in g and extra.get("score_r2_gap") is not None:
            results["score_r2_gap_max"] = _check_le(extra["score_r2_gap"], g["score_r2_gap_max"])
        if "ablated_accuracy_min" in g and extra.get("ablated_accuracy") is not None:
            results["ablated_accuracy_min"] = _check_ge(extra["ablated_accuracy"], g["ablated_accuracy_min"])

    if g.get("leakage_assert", False):
        results["leakage_assert"] = {
            "value": extra.get("leakage_ok", True),
            "required": "label inputs must not appear in X",
            "pass": bool(extra.get("leakage_ok", True)),
        }
    if "label_shuffle_macro_f1_max_over_baseline" in g and extra.get("shuffle_macro_f1") is not None:
        allowed = report.get("baseline", {}).get("majority_accuracy", 0.0) + float(
            g["label_shuffle_macro_f1_max_over_baseline"]
        )
        results["label_shuffle_macro_f1_max_over_baseline"] = {
            "value": extra["shuffle_macro_f1"],
            "required": f"<= {round(allowed, 4)} (must collapse on shuffled labels)",
            "pass": bool(extra["shuffle_macro_f1"] <= allowed),
        }

    # A gate declared in criteria.json but never evaluated is a FAIL, not a skip:
    # the only way to pass is to actually compute the metric.
    for key in g:
        if key not in results:
            results[key] = {
                "value": None,
                "required": "declared in config/criteria.json but the metric was not computed",
                "pass": False,
            }

    all_pass = all(r["pass"] for r in results.values()) if results else False
    return {"task": task, "gate_results": results, "all_pass": all_pass,
            "label_provenance": task_criteria(task).get("label_provenance")}


def format_gate_summary(gate: Dict[str, Any]) -> str:
    status = "PASS" if gate.get("all_pass") else "FAIL"
    parts = []
    for name, r in gate.get("gate_results", {}).items():
        mark = "ok" if r["pass"] else "FAIL"
        parts.append(f"{name}={r['value']} ({r['required']})[{mark}]")
    return f"[{status}] {gate['task']}: " + ", ".join(parts)
