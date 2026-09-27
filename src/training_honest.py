"""Honest training loop for the SideQuest India pipeline.

Replaces the training half of ``src/unified_model.py`` (which built labels from
the same columns it trained on) while keeping the same artifact contract:
``models/<task>_model.pkl`` is still a pickle of a scikit-learn-compatible
estimator, so ``UnifiedSideQuestModel.load_models``/``predict`` and the API
servers keep working.

What this loop guarantees
-------------------------
1. Labels come from ``src/labels_honest.py`` only - the legacy in-model label
   generators are never called. A missing target aborts training instead of
   being fabricated.
2. ``assert_no_leakage`` runs before fitting: a label's input columns may never
   appear in X.
3. Features are restricted by ``config/criteria.json`` ``features_excluded_from_X``.
4. Splits: stratified for classification, quantile-bin stratified for
   regression, grouped by state/district where the label is group-level
   (safety). All medians/percentiles are estimated on the train fold only.
5. Class imbalance: ``scale_pos_weight`` for XGBoost, inverse-frequency
   sample weights for EBM, tuned decision thresholds, per-class metrics.
6. A label-shuffle sanity test must collapse before gates are trusted.
7. Gates are read from ``config/criteria.json`` - no threshold lives in code.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split

import evaluation as ev
from labels_honest import (
    GemLabeler,
    assert_no_leakage,
    build_all_labels,
    load_criteria,
    prohibited_features,
    task_criteria,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_FILENAMES = {
    "hidden_gem": "hidden_gem_model.pkl",
    "hidden_gem_class": "hidden_gem_class_model.pkl",
    "safety_score": "safety_score_model.pkl",
    "collaboration_auth": "collaboration_auth_model.pkl",
    "fake_detection": "fake_detection_model.pkl",
    "solo_matching": "solo_matching_model.pkl",
}

CLASSIFICATION_TASKS = {"collaboration_auth", "fake_detection", "solo_matching", "hidden_gem_class"}


def _is_categorical(s: pd.Series) -> bool:
    # pandas >= 3 reads text as the "str" dtype, not object
    return (s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(s.dtype)) and not pd.api.types.is_bool_dtype(s.dtype)


def _fit_medians(df: pd.DataFrame) -> Dict[str, float]:
    """Train-fold medians for numeric columns (used by serving too)."""
    med = df.apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    return {k: (None if pd.isna(v) else float(v)) for k, v in med.items()}


# ---------------------------------------------------------------------------
# split helpers
# ---------------------------------------------------------------------------
def _bin_strata(y: pd.Series, q: int = 10) -> np.ndarray:
    y = pd.Series(y).reset_index(drop=True)
    nunique = y.nunique()
    if nunique < 2:
        return np.zeros(len(y), dtype=int)
    bins = pd.qcut(y, q=min(q, int(nunique)), labels=False, duplicates="drop")
    return bins.fillna(0).astype(int).values


def group_stratified_split(
    groups: pd.Series,
    y: pd.Series,
    frac: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split whole groups into (keep, held-out) with matched label distributions.

    Safety labels are constant within a district and 99.7% of their variance is
    *between* groups, so a plain GroupShuffleSplit produces folds with wildly
    different label spreads (we measured train std 18.8 / val 27.4 / test 15.0)
    - any model then looks like it memorises, because the folds are not
    comparable. Here groups are binned by their own mean label and a fixed
    share of every bin is held out, so all folds share the same distribution.
    """
    g = pd.Series(groups).reset_index(drop=True)
    tgt = pd.Series(y).reset_index(drop=True)
    frame = pd.DataFrame({"g": g.to_numpy(), "y": tgt.to_numpy()})
    gm = frame.groupby("g")["y"].mean()
    if len(gm) < 8:
        idx = np.arange(len(tgt))
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - frac))
        return idx[:cut], idx[cut:]
    try:
        bins = pd.qcut(gm, q=5, duplicates="drop")
    except Exception:
        bins = pd.Series(0, index=gm.index)
    rng = np.random.default_rng(random_state)
    held: List[Any] = []
    # stratify by GROUP COUNT per label-bin (every bin contributes the same
    # share of its districts). Rows cannot be balanced as well: 36 state-level
    # groups carry ~65% of all rows and a group may never be split, so the
    # primary metric for these tasks is group-level R^2, not row-level.
    for _, members in gm.groupby(bins):
        arr = list(members.index)
        rng.shuffle(arr)
        held.extend(arr[: max(1, int(round(len(arr) * frac)))])
    is_held = g.isin(set(held)).to_numpy()
    return np.where(~is_held)[0], np.where(is_held)[0]


def make_splits(
    y: pd.Series,
    task: str,
    groups: Optional[pd.Series] = None,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the *training pool* into (fit, validation) positionally.

    Classification is stratified on the class label, regression on quantile
    bins of the target, and any group-level label (safety) keeps whole
    districts on one side. Test rows are handled outside by the outer split.
    """
    idx = np.arange(len(y))

    if groups is not None and len(pd.unique(groups)) >= 8:
        keep, val = group_stratified_split(
            groups, y, frac=val_fraction, random_state=random_state
        )
        return np.asarray(keep, dtype=int), np.asarray(val, dtype=int)

    if task in CLASSIFICATION_TASKS:
        strat = y.values
    else:
        strat = _bin_strata(y)

    if len(np.unique(strat)) > 1:
        a, b = train_test_split(idx, test_size=val_fraction, random_state=random_state, stratify=strat)
        return np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    a, b = train_test_split(idx, test_size=val_fraction, random_state=random_state)
    return np.asarray(a, dtype=int), np.asarray(b, dtype=int)


# ---------------------------------------------------------------------------
# model factory
# ---------------------------------------------------------------------------
def make_model(task: str, crit: Dict[str, Any], config: Dict[str, Any], n_classes: int = 2):
    """Build the estimator declared in criteria.json for this task."""
    model_type = crit.get("model_type", "xgboost")
    is_cls = task in CLASSIFICATION_TASKS
    rs = int(config.get("random_state", 42))

    if model_type == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

        common = dict(
            random_state=rs,
            max_bins=int(config.get("ebm_max_bins", 32)),
            interactions=int(config.get("ebm_interactions", 0)),
            outer_bags=int(config.get("ebm_outer_bags", 4)),
            early_stopping_rounds=int(config.get("ebm_early_stopping", 10)),
            min_samples_leaf=int(config.get("ebm_min_samples_leaf", 4)),
            max_interaction_bins=int(config.get("ebm_max_interaction_bins", 32)),
        )
        return ExplainableBoostingClassifier(**common) if is_cls else ExplainableBoostingRegressor(**common)

    if model_type in ("xgboost", "xgboost_text"):
        import xgboost as xgb

        depth = int(config.get("max_depth", 8))
        params = dict(
            n_estimators=int(config.get("n_estimators", 800)),
            max_depth=depth,
            learning_rate=float(config.get("learning_rate", 0.03)),
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=int(config.get("min_child_weight", 5)),
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=rs,
            n_jobs=int(config.get("n_jobs", -1)),
        )
        if is_cls:
            params["eval_metric"] = "logloss"
            esr = int(config.get("early_stopping_rounds", 30))
            if esr > 0:
                params["early_stopping_rounds"] = esr
            params["scale_pos_weight"] = float(config.get("scale_pos_weight", 1.0))
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)

    raise ValueError(f"unknown model_type {model_type!r} for task {task!r}")


def _clone_model(model):
    from sklearn.base import clone

    return clone(model)


def imbalance_weights(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Inverse-frequency sample weights (used by EBM, which has no
    scale_pos_weight). Returns (weights, info)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.ones(len(y)), {"n_classes": 1}
    freq = {c: n for c, n in zip(classes, counts)}
    w = np.array([len(y) / (len(classes) * freq[v]) for v in y], dtype=float)
    info = {
        "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
        "ratio_majority_over_minority": round(float(max(counts) / max(min(counts), 1)), 4),
        "weights": {str(c): round(float(len(y) / (len(classes) * n)), 4) for c, n in zip(classes, counts)},
    }
    return w, info


def scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 1.0
    # positive class = minority class (the one we care about missing)
    maj, mn = float(counts.max()), float(counts.min())
    return round(maj / max(mn, 1.0), 4)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
class HonestTrainer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.criteria = load_criteria()
        self.config = dict(config or {})
        self.config.setdefault("random_state", self.criteria.get("random_state", 42))
        self.config.setdefault("test_size", self.criteria["split"]["test_size"])
        self.config.setdefault("val_fraction_of_train", self.criteria["split"]["val_fraction_of_train"])
        self.config.setdefault("max_depth", 6)
        self.config.setdefault("n_estimators", 400)
        self.config.setdefault("min_child_weight", 8)
        self.models: Dict[str, Any] = {}
        self.reports: Dict[str, Dict[str, Any]] = {}
        self.gates: Dict[str, Dict[str, Any]] = {}
        self.features: Dict[str, List[str]] = {}
        self.medians: Dict[str, Dict[str, float]] = {}

    # -- helpers --------------------------------------------------------------
    def _features_for(self, task: str, data: pd.DataFrame) -> List[str]:
        # Feature definitions stay with the task configs; criteria.json is the
        # contract for what must be EXCLUDED (label inputs) and that is enforced.
        from unified_model import UnifiedSideQuestModel

        # hidden_gem_class should use the SAME base features as hidden_gem
        base_task = "hidden_gem" if task == "hidden_gem_class" else task

        # Try to get base features from UnifiedSideQuestModel
        try:
            base = list(UnifiedSideQuestModel()._setup_task_configs()[base_task].features)
        except KeyError:
            # Task not in unified model - use all numeric/categorical columns
            # except prohibited ones and obvious non-features
            excluded = set(prohibited_features(task))
            excluded.update({"place_id", "name", "city", "state", "area", "address",
                           "source", "categories", "primary_category", "rest_type",
                           "hidden_gem_score", "safety_score", "is_authentic", "is_gem",
                           "ncrb_group", "_outer_split"})
            base = [c for c in data.columns if c not in excluded]

        banned = set(prohibited_features(task))
        feats = [f for f in base if f in data.columns and f not in banned]
        assert_no_leakage(task, feats)
        missing = sorted(set(base) - set(feats) - banned)
        if missing:
            logger.info("  %s: features dropped (absent or prohibited): %s", task, missing)
        return feats

    # -- single task ----------------------------------------------------------
    def train_task(
        self,
        task: str,
        data: pd.DataFrame,
        shuffle_test: bool = True,
        shuffle_subsample: int = 25000,
    ) -> Dict[str, Any]:
        t0 = time.time()
        crit = self.criteria["tasks"][task]
        target = {
            "hidden_gem": "hidden_gem_score",
            "hidden_gem_class": "is_gem",
            "safety_score": "safety_score",
            "collaboration_auth": "is_authentic",
            "fake_detection": "is_fake",
            "solo_matching": "match_label",
        }[task]

        # labels + outer split for this task, constants fitted on its train rows
        if "_outer_split" not in data.columns or target not in data.columns:
            data, split_info = build_task_frame(task, data)
        else:
            split_info = {"split": "prebuilt"}

        if target not in data.columns:
            raise RuntimeError(
                f"target {target!r} for {task} is missing - labels must be built by "
                "src/labels_honest.py, never fabricated inside the trainer"
            )

        feats = self._features_for(task, data)
        if not feats:
            raise RuntimeError(f"no usable features for {task}")

        y = data[target]
        mask = y.notna()
        y = y[mask]
        if task in CLASSIFICATION_TASKS:
            # labels may arrive as 0.0/1.0 (NaN-able floats); estimators such as
            # EBM then report string/float classes_ and P(positive) gets picked
            # from the wrong column
            y = y.astype(int)
        if len(y) < 200:
            raise RuntimeError(f"only {len(y)} labelled rows for {task} - too few to train")

        # The outer train/test split was fixed BEFORE labels were fitted
        # (build_task_frame) so no test row ever influenced a label's
        # cross-row constants. Here we only carve the validation fold out of
        # the training pool.
        if "_outer_split" in data.columns:
            outer = data.loc[mask, "_outer_split"]
            tr_pool = y.index[outer == "train"]
            te_rows = y.index[outer == "test"]
        else:
            tr_pool, te_rows = y.index, y.index[:0]

        groups = None
        if crit.get("group_by"):
            gcol = crit["group_by"]
            if gcol in data.columns:
                groups = data.loc[tr_pool, gcol]

        tr_i, va_i = make_splits(
            y.loc[tr_pool].reset_index(drop=True),
            task,
            groups=groups.reset_index(drop=True) if groups is not None else None,
            val_fraction=float(self.config["val_fraction_of_train"]),
            random_state=int(self.config["random_state"]),
        )
        tr_i = tr_pool.to_numpy()[tr_i]
        va_i = tr_pool.to_numpy()[va_i]
        te_i = te_rows.to_numpy() if len(te_rows) else tr_i[:0]

        # ---- design matrix: all transforms fit on the FIT fold only ----
        num_feats = [f for f in feats if not _is_categorical(data[f])]
        cat_feats = [f for f in feats if _is_categorical(data[f])]
        med = _fit_medians(data.loc[tr_i, num_feats])
        cat_maps = {}
        for f in cat_feats:
            seen = data.loc[tr_i, f].dropna().astype(str).unique().tolist()
            cat_maps[f] = {v: i for i, v in enumerate(seen)}

        def design(frame: pd.DataFrame) -> pd.DataFrame:
            out = pd.DataFrame(index=frame.index)
            for f in num_feats:
                s = pd.to_numeric(frame[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
                out[f] = s.fillna(med.get(f, 0.0))
            for f in cat_feats:
                out[f] = frame[f].astype(str).map(cat_maps[f]).fillna(-1).astype(float)
            return out[feats].astype(float)  # same column order as feature_specs.json

        Xtr, Xva, Xte = design(data.loc[tr_i]), design(data.loc[va_i]), design(data.loc[te_i])
        ytr, yva, yte = y.loc[tr_i], y.loc[va_i], y.loc[te_i]

        is_cls = task in CLASSIFICATION_TASKS
        run_cfg = dict(self.config)
        imbalance_info: Dict[str, Any] = {}
        sample_weight = None

        if is_cls:
            spw = scale_pos_weight(ytr.values)
            run_cfg["scale_pos_weight"] = spw
            w, imbalance_info = imbalance_weights(ytr.values)
            imbalance_info["scale_pos_weight"] = spw
            if crit.get("model_type") == "ebm":
                sample_weight = w
        else:
            _, imbalance_info = imbalance_weights(_bin_strata(ytr))
            imbalance_info["note"] = "regression target stratified by quantile bins"

        # per-task model settings from criteria.json (e.g. EBM interactions)
        run_cfg.update(crit.get("model_params", {}))
        model = make_model(task, crit, run_cfg)
        fit_kw: Dict[str, Any] = (
            {"sample_weight": sample_weight} if sample_weight is not None else {}
        )

        # XGBoost: train to the cap, then trim to the validation-optimal tree
        # count (xgboost's own early_stopping_rounds proved unreliable here).
        # EBM: its internal bagging/early stopping already regularises.
        if type(model).__name__.startswith("XGB"):
            model = _fit_xgb(
                model, Xtr, ytr, Xva, yva,
                "classification" if is_cls else "regression",
                sample_weight=sample_weight,
            )
        else:
            model.fit(Xtr, ytr, **fit_kw)

        # ---- reports ----
        if is_cls:
            thr_spec = crit.get("decision_threshold", {})
            threshold = float(thr_spec.get("value", 0.5))
            counts = {c: int((ytr.values == c).sum()) for c in np.unique(ytr)}
            minority = min(counts, key=counts.get)
            pos = crit.get("positive_class_value", minority)
            if pos not in counts:
                pos = minority

            score_va = _predict_proba_for(model, Xva, pos)
            tuned = None
            if thr_spec.get("tune_on_validation") and score_va is not None:
                tuned = ev.tune_threshold(
                    yva.values, score_va,
                    thr_spec.get("optimise_for", "minority_f1"), positive=pos,
                )
                threshold = tuned["threshold"]

            score_te = _predict_proba_for(model, Xte, pos)
            score_tr = _predict_proba_for(model, Xtr, pos)
            other = next((c for c in sorted(set(ytr.tolist())) if c != pos), pos)
            y_pred_te = (
                np.where(score_te >= threshold, pos, other)
                if score_te is not None else model.predict(Xte)
            )
            y_pred_tr = (
                np.where(score_tr >= threshold, pos, other)
                if score_tr is not None else model.predict(Xtr)
            )
            test_report = ev.classification_report(yte.values, y_pred_te, score_te, positive=pos)
            train_report = ev.classification_report(ytr.values, y_pred_tr, score_tr, positive=pos)
            test_report["decision_threshold"] = threshold
            test_report["threshold_tuning"] = tuned
            extra = {
                "accuracy_gap": max(0.0, train_report["accuracy"] - test_report["accuracy"]),
                "leakage_ok": True,
            }
        else:
            pred_te = model.predict(Xte)
            pred_tr = model.predict(Xtr)
            gcol = crit.get("group_by")
            grp_tr = data.loc[tr_i, gcol] if gcol and gcol in data.columns else None
            grp_te = data.loc[te_i, gcol] if gcol and gcol in data.columns else None
            test_report = ev.regression_report(yte.values, pred_te, groups=grp_te)
            train_report = ev.regression_report(ytr.values, pred_tr, groups=grp_tr)
            test_report["r2_gap"] = max(0.0, train_report["r2"] - test_report["r2"])
            extra = {"r2_gap": test_report["r2_gap"], "leakage_ok": True}
            if "group_r2" in test_report:
                test_report["group_r2_gap"] = max(
                    0.0, train_report.get("group_r2", 0.0) - test_report["group_r2"]
                )
                extra["group_r2_gap"] = test_report["group_r2_gap"]

        # ---- label-shuffle sanity test ----
        shuffle_res = None
        if shuffle_test:
            sub = min(shuffle_subsample, len(Xtr))
            rng = np.random.default_rng(int(self.config["random_state"]))
            take = rng.choice(len(Xtr), size=sub, replace=False)
            Xs, ys = Xtr.iloc[take], ytr.iloc[take].copy()
            shuffled = ys.to_numpy().copy()
            rng.shuffle(shuffled)
            try:
                s_cfg = dict(run_cfg)
                s_cfg["n_estimators"] = int(run_cfg.get("n_estimators", 400)) // 2
                s_cfg["early_stopping_rounds"] = 0
                s_cfg["ebm_outer_bags"] = 1
                clone = make_model(task, crit, s_cfg)
                clone.fit(Xs, pd.Series(shuffled))
                if is_cls:
                    sc = {c: int((shuffled == c).sum()) for c in np.unique(shuffled)}
                    pos_s = min(sc, key=sc.get)
                    other_s = next((c for c in sc if c != pos_s), pos_s)
                    # judged on HELD-OUT validation rows with their true labels: a model
                    # that learnt only noise must be no better than chance there
                    p = _predict_proba_for(clone, Xva, pos_s)
                    pr = (
                        np.where(p >= 0.5, pos_s, other_s)
                        if p is not None else clone.predict(Xva)
                    )
                    rep = ev.classification_report(yva.to_numpy(), pr, p, positive=pos_s)
                    shuffle_res = {
                        "metric": "macro_f1",
                        "shuffled_value": rep["macro_f1"],
                        "baseline": rep["baseline"]["majority_accuracy"],
                        "collapsed": bool(
                            rep["macro_f1"] <= rep["baseline"]["majority_accuracy"] + 0.10
                        ),
                    }
                    extra["shuffle_macro_f1"] = rep["macro_f1"]
                else:
                    # held-out R2 of a model trained on shuffled labels (must be ~0);
                    # the in-sample value only shows how much a model can memorise
                    rep = ev.regression_report(yva.to_numpy(), clone.predict(Xva))
                    memo = ev.regression_report(shuffled, clone.predict(Xs))["r2"]
                    shuffle_res = {
                        "metric": "r2 (held-out, true labels)",
                        "shuffled_value": rep["r2"],
                        "in_sample_memorisation_r2": memo,
                        "baseline": 0.0,
                        "collapsed": bool(rep["r2"] <= 0.01),
                    }
                    extra["shuffle_r2"] = rep["r2"]
            except Exception as exc:
                shuffle_res = {"error": str(exc), "collapsed": None}
                logger.warning("shuffle test failed for %s: %s", task, exc)

        # ---- gates ----
        gate = ev.evaluate_gates(task, test_report, extra)
        gate["shuffle_test"] = shuffle_res
        gate["label_provenance"] = crit.get("label_provenance")

        result = {
            "task": task,
            "model_type": crit.get("model_type"),
            "label_provenance": crit.get("label_provenance"),
            "label_source": crit.get("label_source"),
            "n_samples": int(len(y)),
            "n_train": int(len(tr_i)),
            "n_val": int(len(va_i)),
            "n_test": int(len(te_i)),
            "n_features": len(feats),
            "features": feats,
            "split": split_info.get("split"),
            "outer_split": split_info,
            "inner_split": "grouped" if groups is not None else ("stratified" if is_cls else "quantile_bin_stratified"),
            "class_balance": imbalance_info,
            "threshold": test_report.get("decision_threshold"),
            "report": test_report,
            "train_report": train_report,
            "gates": gate,
            "duration_s": round(time.time() - t0, 1),
        }

        self.models[task] = model
        self.reports[task] = result
        self.gates[task] = gate
        self.features[task] = feats
        self.medians[task] = {"numeric": med, "categorical": cat_maps}
        return result

    # -- persistence ----------------------------------------------------------
    def save(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        for task, model in self.models.items():
            path = MODELS_DIR / MODEL_FILENAMES.get(task, f"{task}_model.pkl")
            with open(path, "wb") as fh:
                pickle.dump(model, fh)
            logger.info("  saved %s", path.name)

        def _merged(name: str, new: Dict[str, Any]) -> Dict[str, Any]:
            """Update a per-task json file instead of replacing it, so a partial
            run never drops the serving contract of tasks it did not retrain."""
            path = MODELS_DIR / name
            try:
                cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except ValueError:
                cur = {}
            cur = {k: v for k, v in cur.items() if k != "safety_score"}
            cur.update(new)
            return cur

        (MODELS_DIR / "features.json").write_text(
            json.dumps(_merged("features.json", self.features), indent=2), encoding="utf-8"
        )
        (MODELS_DIR / "imputer_medians.json").write_text(
            json.dumps(_merged("imputer_medians.json", self.medians), indent=2), encoding="utf-8"
        )
        # serving contract: exact feature order, categorical maps and medians,
        # all fitted on the training fold - inference.py reads this file
        specs = {}
        for task in self.features:
            specs[task] = {
                "features": self.features[task],
                "numeric_medians": self.medians[task].get("numeric", {}),
                "categorical_maps": self.medians[task].get("categorical", {}),
                "decision_threshold": (
                    self.reports[task]["report"].get("decision_threshold")
                    if task in self.reports else None
                ),
                "model_type": self.criteria["tasks"].get(task, {}).get("model_type"),
            }
        (MODELS_DIR / "feature_specs.json").write_text(
            json.dumps(_merged("feature_specs.json", specs), indent=2), encoding="utf-8"
        )
        (MODELS_DIR / "criteria_snapshot.json").write_text(
            json.dumps(self.criteria, indent=2), encoding="utf-8"
        )
        self.write_reports()

    def write_reports(self) -> None:
        summary: Dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "criteria_version": self.criteria.get("version"),
            "tasks": {},
        }
        gates_out: Dict[str, Any] = {"all_gates_pass": True, "gates": {}}

        # Keep results of tasks NOT retrained in this run (a partial run used to
        # wipe them from the reports, which is how safety "disappeared").
        # Tasks that are no longer trained at all (safety_score -> index) are dropped.
        trainable = {t for t, c in self.criteria["tasks"].items() if c.get("type") != "index"}
        prev_m = REPORTS_DIR / "metrics.json"
        prev_g = REPORTS_DIR / "gates.json"
        if prev_m.exists():
            try:
                old_tasks = json.loads(prev_m.read_text(encoding="utf-8")).get("tasks", {})
                old_gates = json.loads(prev_g.read_text(encoding="utf-8")).get("gates", {}) \
                    if prev_g.exists() else {}
                for t, rep in old_tasks.items():
                    if t in trainable and t not in self.reports:
                        rep = dict(rep)
                        rep["carried_over_from"] = rep.get("generated_at") or \
                            json.loads(prev_m.read_text(encoding="utf-8")).get("generated_at")
                        summary["tasks"][t] = rep
                        if t in old_gates:
                            gates_out["gates"][t] = old_gates[t]
                            gates_out["all_gates_pass"] = (
                                gates_out["all_gates_pass"] and bool(old_gates[t].get("all_pass")))
            except (ValueError, OSError) as exc:
                logger.warning("could not merge previous reports (%s) - writing fresh", exc)

        for task, res in self.reports.items():
            rep = dict(res["report"])
            rep.update(
                {
                    "type": res["report"]["type"],
                    "model_type": res["model_type"],
                    "label_provenance": res["label_provenance"],
                    "label_source": res["label_source"],
                    "n_train": res["n_train"],
                    "n_val": res["n_val"],
                    "n_test": res["n_test"],
                    "n_features": res["n_features"],
                    "split": res["split"],
                    "outer_split": res.get("outer_split"),
                    "inner_split": res.get("inner_split"),
                    "class_balance": res["class_balance"],
                    "decision_threshold": res["threshold"],
                    "train_r2" if res["report"]["type"] == "regression" else "train_accuracy": (
                        res["train_report"].get("r2") if res["report"]["type"] == "regression"
                        else res["train_report"].get("accuracy")
                    ),
                    "gate_results": res["gates"]["gate_results"],
                    "gate_pass": res["gates"]["all_pass"],
                    "shuffle_test": res["gates"].get("shuffle_test"),
                }
            )
            summary["tasks"][task] = rep
            gates_out["gates"][task] = res["gates"]
            gates_out["all_gates_pass"] = gates_out["all_gates_pass"] and res["gates"]["all_pass"]

        (REPORTS_DIR / "metrics.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        (REPORTS_DIR / "gates.json").write_text(
            json.dumps(gates_out, indent=2, default=str), encoding="utf-8"
        )

        lines = [
            "",
            "=" * 62,
            "SIDEQUEST MODEL SUMMARY (honest labels, criteria.json gates)",
            "=" * 62,
            f"Generated: {summary['generated_at']}",
            "",
        ]
        for task, rep in summary["tasks"].items():
            prov = rep.get("label_provenance", "?")
            lines.append(f"{task.upper()}  [{rep.get('model_type')}]  label={prov}")
            if rep["type"] == "regression":
                def _f(v, fmt):
                    return "n/a" if v is None else format(v, fmt)
                lines.append(
                    f"  R2={rep['r2']:.4f} (margin over baseline {_f(rep.get('r2_minus_baseline'), '.4f')})  "
                    f"RMSE={_f(rep.get('rmse'), '.4f')}  MAE={_f(rep.get('mae'), '.4f')}"
                )
                lines.append(
                    f"  target: mean={_f(rep.get('target_mean'), '.2f')} std={_f(rep.get('target_std'), '.2f')} "
                    f"n={rep.get('n_samples')}"
                )
            else:
                lines.append(
                    f"  accuracy={rep['accuracy']:.4f} (majority baseline "
                    f"{rep['baseline']['majority_accuracy']:.4f})  "
                    f"macro_f1={rep['macro_f1']:.4f}  balanced_acc={rep['balanced_accuracy']:.4f}"
                )
                lines.append(
                    f"  minority(class {rep['minority_class']}): recall={rep['minority_recall']:.4f} "
                    f"f1={rep['minority_f1']:.4f}   pr_auc={rep.get('pr_auc', 0):.4f} "
                    f"(lift x{rep.get('pr_auc_lift_over_prevalence', 0):.2f})"
                )
                lines.append(f"  confusion matrix {rep['confusion_matrix_labels']}: {rep['confusion_matrix']}")
            g = rep.get("gate_results", {})
            mark = "PASS" if rep.get("gate_pass") else "FAIL"
            lines.append(f"  GATES [{mark}]: " + ", ".join(
                f"{k}={v['value']}({v['required']})" for k, v in g.items()
            ))
            lines.append("")
        lines.append(f"OVERALL: {'ALL GATES PASS' if gates_out['all_gates_pass'] else 'SOME GATES FAILED'}")
        lines.append("=" * 62)
        (REPORTS_DIR / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        logger.info("reports written to %s", REPORTS_DIR)


# ---------------------------------------------------------------------------
def _select_trees(model, X_val, y_val, cap: int, step: int = 50, task: str = "regression"):
    """Anti-overfit control: pick the number of boosting rounds by validation
    performance, then refit a fresh model with exactly that many.

    xgboost's own ``early_stopping_rounds`` fired at iteration 0 on this data
    (validation RMSE never beat round 1 within its patience window), which
    silently produced a 1-tree model. Scanning the validation curve ourselves is
    both correct and cheaper than trusting the constructor flag.
    """
    import xgboost as xgb

    bst = model.get_booster()
    dval = xgb.DMatrix(X_val)
    yv = np.asarray(y_val, dtype=float)
    best_k, best_score = cap, float("inf")
    for k in range(step, cap + 1, step):
        raw = bst.predict(dval, iteration_range=(0, k))
        if task == "classification":
            p = np.clip(raw, 1e-6, 1 - 1e-6)
            # log loss for the positive class; lower is better
            score = float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
        else:
            score = float(np.sqrt(np.mean((raw - yv) ** 2)))
        if score < best_score - 1e-6:
            best_score, best_k = score, k
    return best_k, best_score


def _fit_xgb(model, X_tr, y_tr, X_va, y_va, task: str, sample_weight=None):
    """Fit an XGBoost estimator, trim it to the validation-optimal tree count,
    refit once at that size, and record the choice on the model."""
    params = model.get_xgb_params()
    cap = int(params.get("n_estimators") or 400)
    params.pop("early_stopping_rounds", None)
    probe_cls = model.__class__
    probe = probe_cls(**{**params, "n_estimators": cap})
    kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
    probe.fit(X_tr, y_tr, **kw)
    best_k, best_score = _select_trees(probe, X_va, y_va, cap, task=task)
    final = probe_cls(**{**params, "n_estimators": int(best_k)})
    final.fit(X_tr, y_tr, **kw)
    final.n_estimators_used_ = int(best_k)
    final.validation_score_ = float(best_score)
    logger.info("  %s: trees=%d/%d (validation %s=%.4f)",
                type(final).__name__, best_k, cap,
                "logloss" if task == "classification" else "rmse", best_score)
    return final


def _predict_proba_for(model, X, label) -> Optional[np.ndarray]:
    """P(label) for any binary label value, or None when the estimator has no
    usable score. Every probability in the reports is for the class of
    interest (the minority/positive class), never for the majority."""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = list(getattr(model, "classes_", [0, 1]))
            if label in classes:
                return np.asarray(proba[:, classes.index(label)], dtype=float)
            return np.asarray(proba[:, 0], dtype=float)
        if hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(X), dtype=float)
            p1 = 1.0 / (1.0 + np.exp(-np.clip(raw, -35, 35)))
            return p1 if label == 1 else 1.0 - p1
    except Exception as exc:
        logger.debug("predict_proba unavailable: %s", exc)
    return None


def _predict_proba_positive(model, X, y_fit=None) -> Optional[np.ndarray]:
    return _predict_proba_for(model, X, 1)


# ---------------------------------------------------------------------------
def prepare_training_frame() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """india_places.csv with district crime rates attached (no labels yet).

    ``SafetyLabeler.enrich_crime_with_district`` is a pure lookup - it fits no
    statistics - so it may run before any split. Labels are built later, per
    task, inside :func:`build_task_frame`, because each task needs its own
    outer split and must fit its own label constants on ITS train fold.
    """
    from labels_honest import PLACES_CSV, SafetyLabeler

    places = pd.read_csv(PLACES_CSV)
    logger.info("loaded %d places", len(places))
    # india_places.csv still carries the OLD heuristic targets written by
    # data_pipeline_india.py (hidden_gem_score = f(competitor_density,
    # dist_from_center, category_rarity, ...), i.e. a formula over model
    # features). They must never reach a trainer - hidden_gem_class used to
    # pick this column up and learned the formula back. Labels are rebuilt from
    # src/labels_honest.py for every task.
    legacy = [c for c in ("hidden_gem_score", "safety_score", "is_gem", "is_authentic")
              if c in places.columns]
    if legacy:
        logger.info("dropping legacy target columns from the input frame: %s", legacy)
        places = places.drop(columns=legacy)
    from gem_features import add_gem_features, dedupe_listings

    n0 = len(places)
    places = dedupe_listings(places).reset_index(drop=True)
    logger.info("dedupe: %d listing rows -> %d unique places", n0, len(places))
    places = add_gem_features(places)
    places = SafetyLabeler.enrich_crime_with_district(places)
    return places, {"n_places": int(len(places))}


def _strat_proxy(task: str, places: pd.DataFrame) -> np.ndarray:
    """Label-independent strata used to split BEFORE labels are computed.

    Stratifying on a proxy of the eventual target (review volume, fake-review
    fraction) is sampling only - it does not leak anything into X.
    """
    if task in ("hidden_gem", "hidden_gem_class"):  # identical split for both
        s = pd.to_numeric(places.get("review_count"), errors="coerce").fillna(0)
        return _bin_strata(s, q=5)
    if task == "collaboration_auth":
        stats = _scored_review_stats()
        if stats is None:
            return np.zeros(len(places), dtype=int)
        j = places[["place_id"]].join(stats["fraud_rate"], on="place_id")
        return _bin_strata(j["fraud_rate"].fillna(-1.0), q=5)
    # safety: stratified inside make_splits by grouping on the district
    return np.zeros(len(places), dtype=int)


def _scored_review_stats() -> Optional[pd.DataFrame]:
    from labels_honest import SCORED_REVIEWS_CSV

    if not SCORED_REVIEWS_CSV.exists():
        logger.warning("%s missing - collaboration_auth labels unavailable", SCORED_REVIEWS_CSV.name)
        return None
    cols = pd.read_csv(SCORED_REVIEWS_CSV, nrows=0).columns
    use = [c for c in ("place_id", "is_fake") if c in cols]
    if "place_id" not in use:
        return None
    s = pd.read_csv(SCORED_REVIEWS_CSV, usecols=use)
    if "is_fake" not in s.columns:
        return None
    return s.groupby("place_id").agg(fraud_rate=("is_fake", "mean"), rev_n=("is_fake", "size"))


def attach_review_features(places: pd.DataFrame) -> pd.DataFrame:
    """Per-place review-set statistics used by collaboration_auth.

    Derived from review text/stars only - never from the fake verdict
    (``p_fake``/``is_fake``/``fraud_rate``), which stays a label input and is
    excluded from X by ``config/criteria.json``. Serving computes exactly the
    same columns for a single place_id.
    """
    from labels_honest import REVIEWS_CSV

    if not REVIEWS_CSV.exists():
        logger.warning("%s missing - review features unavailable", REVIEWS_CSV.name)
        return places
    rev = pd.read_csv(REVIEWS_CSV, usecols=["place_id", "stars", "review_text"])
    rev["stars"] = pd.to_numeric(rev["stars"], errors="coerce")
    rev["words"] = rev["review_text"].fillna("").str.split().str.len()
    agg = rev.groupby("place_id").agg(
        rev_n=("stars", "size"),
        rev_stars_mean=("stars", "mean"),
        rev_stars_std=("stars", "std"),
        rev_avg_words=("words", "mean"),
    )
    agg["rev_stars_std"] = agg["rev_stars_std"].fillna(0.0)

    out = places.join(agg, on="place_id", how="left")
    if "stars" in out.columns:
        out["rev_rating_mismatch"] = (
            pd.to_numeric(out["stars"], errors="coerce")
            - pd.to_numeric(out["rev_stars_mean"], errors="coerce")
        ).abs()
    return out


def build_task_frame(task: str, places: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Label one task and fix its outer train/test split.

    Order of operations is the whole point:
      1. work out which rows are *eligible* for this task (data availability),
      2. split those rows into train/test - grouped by district for safety,
         stratified by a label proxy otherwise,
      3. fit every cross-row label constant on the TRAIN rows only,
      4. transform the full frame (train and test see identical constants).

    Returns (frame with ``<target>`` and ``_outer_split``, split info).
    """
    from labels_honest import (
        GemLabeler,
        REVIEWS_CSV,
        SCORED_REVIEWS_CSV,
        SafetyLabeler,
        build_collab_labels,
        task_criteria,
    )

    crit = task_criteria(task)
    seed = int(load_criteria().get("random_state", 42))
    ts = float(load_criteria()["split"]["test_size"])
    out = places.copy()

    if task == "hidden_gem":
        eligible = pd.Series(False, index=out.index)
        if REVIEWS_CSV.exists():
            end = GemLabeler.endorsement_components(pd.read_csv(REVIEWS_CSV))
            elig_ids = set(end.index[end["n_reviews"] >= int(crit["label_constants"].get("min_reviews", 3))])
            eligible = out["place_id"].isin(elig_ids)
        eligible_idx = out.index[eligible]
        split_kind = "stratified_row_level (single-city label)"
        groups = None

    elif task == "hidden_gem_class":
        # SAME eligible rows and split as hidden_gem - reuse the regression's outer split
        eligible = pd.Series(False, index=out.index)
        if REVIEWS_CSV.exists():
            end = GemLabeler.endorsement_components(pd.read_csv(REVIEWS_CSV))
            gem_min = int(task_criteria("hidden_gem")["label_constants"].get("min_reviews", 1))
            elig_ids = set(end.index[end["n_reviews"] >= gem_min])
            eligible = out["place_id"].isin(elig_ids)
        eligible_idx = out.index[eligible]
        split_kind = "stratified_row_level (single-city label, shared with hidden_gem)"
        groups = None

    elif task == "collaboration_auth":
        stats = _scored_review_stats()
        if stats is None:
            raise RuntimeError("scored reviews unavailable - cannot build collaboration_auth labels")
        joined = out[["place_id"]].join(stats["fraud_rate"], on="place_id")
        eligible_idx = out.index[joined["fraud_rate"].notna()]
        split_kind = "stratified_row_level (proxy = fraud_rate)"
        groups = None

    elif task == "safety_score":
        eligible_idx = out.index
        split_kind = "grouped_by_ncrb_group"
        groups = out["ncrb_group"].fillna(out["state"])
    else:
        raise ValueError(f"no outer-split rule for task {task!r}")

    if task == "collaboration_auth":
        out = attach_review_features(out)

    n = len(eligible_idx)
    if n < 200:
        raise RuntimeError(f"only {n} eligible rows for {task} - too few to train")

    if groups is not None and len(pd.unique(groups.loc[eligible_idx])) >= 8:
        # stratify groups by their pre-label crime proxy so train/test share the
        # same district-level crime distribution (the label itself is not
        # available yet - its decay scales are fitted on the train fold)
        proxy = pd.to_numeric(out.loc[eligible_idx, "violent_crime_rate"], errors="coerce")
        proxy = proxy.fillna(proxy.median())
        a, b = group_stratified_split(
            groups.loc[eligible_idx], proxy, frac=ts, random_state=seed
        )
        tr_rows = eligible_idx[a]
        te_rows = eligible_idx[b]
    else:
        proxy = _strat_proxy(task, out.loc[eligible_idx])
        tr_rows, te_rows = train_test_split(
            eligible_idx, test_size=ts, random_state=seed,
            stratify=proxy if len(np.unique(proxy)) > 1 else None,
        )

    train_rows = out.loc[tr_rows]

    # ---- fit label constants on TRAIN rows only, then transform everything ----
    if task == "hidden_gem":
        lab = GemLabeler()
        if REVIEWS_CSV.exists():
            end = GemLabeler.endorsement_components(pd.read_csv(REVIEWS_CSV))
            lab.fit(train_rows, end)
            out["hidden_gem_score"] = lab.transform(out, end)
        else:
            out["hidden_gem_score"] = np.nan

    if task == "hidden_gem_class":
        # Derive binary label from hidden_gem_score using threshold from criteria
        threshold = crit.get("label_constants", {}).get("threshold", 60.0)
        # ALWAYS rebuild the quality label from reviews, fitted on this split's
        # train rows. Never reuse a hidden_gem_score column found in the input.
        out = out.drop(columns=["hidden_gem_score"], errors="ignore")
        lab = GemLabeler()
        if REVIEWS_CSV.exists():
            end = GemLabeler.endorsement_components(pd.read_csv(REVIEWS_CSV))
            lab.fit(train_rows, end)
            out["hidden_gem_score"] = lab.transform(out, end)
        else:
            out["hidden_gem_score"] = np.nan
        # NaN (too few reviews) stays unlabelled instead of silently becoming 0
        out["is_gem"] = np.where(out["hidden_gem_score"].notna(),
                                 (out["hidden_gem_score"] >= threshold).astype(float), np.nan)

    # (this used to be a separate if/else, so the hidden_gem task ALSO ran the
    # collaboration_auth labeler and needed its data file)
    elif task == "safety_score":
        raise ValueError("safety_score is a rule-based index (src/safety_index.py), not a trained task")
    elif task == "collaboration_auth":  # threshold label, no fitted constants
        scored = pd.read_csv(SCORED_REVIEWS_CSV) if SCORED_REVIEWS_CSV.exists() else None
        out["is_authentic"] = build_collab_labels(out, scored)
        lab = None

    split_col = pd.Series("unlabelled", index=out.index, dtype=object)
    split_col.loc[tr_rows] = "train"
    split_col.loc[te_rows] = "test"
    out["_outer_split"] = split_col

    info = {
        "task": task,
        "split": split_kind,
        "grouped": groups is not None,
        "test_size": ts,
        "n_eligible": int(n),
        "n_train": int(len(tr_rows)),
        "n_test": int(len(te_rows)),
        "fit_on": "this task's train rows only",
        "label_constants_fit": task in ("hidden_gem", "safety_score", "hidden_gem_class"),
        "label_provenance": crit.get("label_provenance"),
    }
    logger.info(
        "  %s: eligible=%d train=%d test=%d split=%s", task, n, len(tr_rows), len(te_rows), split_kind
    )
    return out, info



def train_all(
    tasks: Optional[List[str]] = None,
    shuffle_test: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> HonestTrainer:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # safety_score is a rule-based index (src/safety_index.py), not a trained model
    tasks = tasks or ["hidden_gem", "hidden_gem_class", "collaboration_auth"]
    if "safety_score" in tasks:
        raise ValueError("safety_score is computed by src/safety_index.py - it is not trained")
    data, info = prepare_training_frame()
    trainer = HonestTrainer(config)

    for task in tasks:
        prov = trainer.criteria["tasks"][task].get("label_provenance")
        logger.info("=" * 60)
        logger.info("TRAINING %s  (label=%s)", task, prov)
        logger.info("=" * 60)
        res = trainer.train_task(task, data, shuffle_test=shuffle_test)
        r = res["report"]
        if r["type"] == "regression":
            logger.info("  R2=%.4f RMSE=%.4f MAE=%.4f n=%d",
                        r["r2"], r["rmse"], r["mae"], r["n_samples"])
        else:
            logger.info("  acc=%.4f (baseline %.4f) macro_f1=%.4f minority_recall=%.4f pr_auc=%.4f",
                        r["accuracy"], r["baseline"]["majority_accuracy"], r["macro_f1"],
                        r["minority_recall"], r.get("pr_auc", 0.0))
        logger.info("  gate: %s", ev.format_gate_summary(res["gates"]))
        logger.info("  features=%d split=%s duration=%.1fs",
                    res["n_features"], res["split"], res["duration_s"])

    trainer.save()
    return trainer


if __name__ == "__main__":
    train_all()
