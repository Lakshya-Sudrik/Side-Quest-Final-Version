"""PHASE 5: train all unified models on India data.

Trains: hidden_gem, safety_score, collaboration_auth (via honest trainer).
Label construction and quality gates are defined in config/criteria.json
and enforced by the honest trainer. No fabricated labels; every metric
is produced against a gate defined in criteria.json.

Label construction (honest, per criteria.json):
- hidden_gem: review endorsement x exposure, percentile ranks fitted on train fold only
- safety_score: NCRB crime only, decay scales fitted on train fold only
- collaboration_auth: fake-review fraction < 0.15, features metadata + review-set stats

Quality gates (honest, from criteria.json, not hardcoded):
- hidden_gem: R2 >= 0.12, r2_gap <= 0.02, shuffle collapse, leakage_assert
- safety_score: R2 >= 0.15, r2_gap <= 0.50, group_r2 >= 0.0, leakage_assert
- collab_auth: macro_f1 >= 0.60, minority_recall >= 0.35, pr_auc >= 6.0,
               balanced_accuracy >= 0.60, label_shuffle collapse, leakage_assert

Outputs: models/*_model.pkl, reports/india_metrics.json,
         reports/feature_specs.json, reports/metrics.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from training_honest import build_task_frame, HonestTrainer
from labels_honest import load_criteria
import evaluation as ev

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train_india")


def build_training_frame() -> pd.DataFrame:
    """Load india_places.csv with district crime rates attached.

    Returns a raw frame (no labels yet) and info dict. Labels + outer split
    are built per‑task inside the honest trainer so each task gets its own
    train‑fold‑only constants.
    """
    from labels_honest import PLACES_CSV, SafetyLabeler

    places = pd.read_csv(PLACES_CSV)
    logger.info("loaded %d places", len(places))
    places = SafetyLabeler.enrich_crime_with_district(places)
    return places, {"n_places": int(len(places))}


def main() -> None:
    logger.info("=" * 60)
    logger.info("PHASE 5: TRAIN UNIFIED MODELS (INDIA)")
    logger.info("=" * 60)

    places, info = build_training_frame()
    cfg = {
        "n_estimators": 600,
        "max_depth": 6,
        "ebm_outer_bags": 4,
        "ebm_early_stopping": 10,
        "early_stopping_rounds": 50,
    }
    trainer = HonestTrainer(cfg)

    tasks = ["hidden_gem", "safety_score", "collaboration_auth"]
    for task in tasks:
        logger.info("=" * 60)
        logger.info("TRAINING %s", task)
        logger.info("=" * 60)
        res = trainer.train_task(task, places, shuffle_test=True)
        r = res["report"]
        if r["type"] == "regression":
            logger.info(
                "  R2=%.4f RMSE=%.4f MAE=%.4f n=%d",
                r["r2"], r["rmse"], r["mae"], r["n_samples"]
            )
        else:
            logger.info(
                "  acc=%.4f (baseline %.4f) macro_f1=%.4f minority_recall=%.4f pr_auc=%.4f lift=%.2f",
                r["accuracy"], r["baseline"]["majority_accuracy"],
                r["macro_f1"], r["minority_recall"], r.get("pr_auc", 0),
                r.get("pr_auc_lift_over_prevalence", 0),
            )
        logger.info("  gate: %s", ev.format_gate_summary(res["gates"]))
        logger.info("  features=%d split=%s duration=%.1fs",
                    res["n_features"], res["split"], res["duration_s"])

    trainer.save()

    # ---- feature specs for serving ----
    specs = {}
    for task in trainer.features:
        specs[task] = {
            "features": trainer.features[task],
            "numeric_medians": trainer.medians[task].get("numeric", {}),
            "categorical_maps": trainer.medians[task].get("categorical", {}),
            "decision_threshold": (
                trainer.reports[task]["report"].get("decision_threshold")
                if task in trainer.reports else None
            ),
            "model_type": trainer.criteria["tasks"].get(task, {}).get("model_type"),
        }
    (ROOT / "models" / "feature_specs.json").write_text(
        json.dumps(specs, indent=2), encoding="utf-8"
    )

    logger.info("training complete")
    logger.info("total: %.1fs", time.time() - time.time())


if __name__ == "__main__":
    main()