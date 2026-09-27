"""
SideQuest Training Script - Full real data with tuned hyperparameters
"""
import os as _sq_os
from pathlib import Path as _SqPath
# project root: override with SIDEQUEST_ROOT, otherwise the repo checkout
_SQ_ROOT = _sq_os.environ.get("SIDEQUEST_ROOT", str(_SqPath(__file__).resolve().parents[0])).replace("\\", "/")

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from src.unified_model import UnifiedSideQuestModel, ModelMetrics
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SideQuestTrainer:
    def __init__(self):
        self.data_dir = Path(f"{_SQ_ROOT}/data")
        self.models_dir = Path(f"{_SQ_ROOT}/models")
        self.reports_dir = Path(f"{_SQ_ROOT}/reports")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.model = UnifiedSideQuestModel()
        self.rag = RAGPipeline()
        self.raw_data = {}
        self.processed_data = None

    def load_real_data(self):
        merged_path = self.data_dir / "merged_places.csv"
        fake_path = self.data_dir / "fake_review_features.csv"

        if not merged_path.exists() or not fake_path.exists():
            logger.info("Pipeline CSVs not found. Running data pipeline...")
            from data_pipeline import run as run_data_pipeline
            run_data_pipeline()

        self.processed_data = pd.read_csv(merged_path)
        logger.info(f"Loaded merged_places: {self.processed_data.shape}")

        self.raw_data["fake_reviews"] = pd.read_csv(fake_path)
        logger.info(f"Loaded fake_review_features: {self.raw_data['fake_reviews'].shape}")

        return self.processed_data

    def train_model(self):
        logger.info("Training unified model on real data...")

        if self.processed_data is None:
            raise ValueError("No data loaded.")

        # Train Tasks 2, 3, 4
        logger.info("\n--- Training Tasks 2, 3, 4 ---")
        results_234 = self.model.train(
            self.processed_data,
            tasks=["hidden_gem", "safety_score", "collaboration_auth"]
        )

        # Train Task 1
        logger.info("\n--- Training Task 1 (fake_detection) ---")
        fake_data = self.raw_data.get("fake_reviews")
        if fake_data is not None:
            results_1 = self.model.train(fake_data, tasks=["fake_detection"])
            results_234.update(results_1)

        return results_234

    def evaluate_model(self):
        logger.info("Evaluating model...")
        summary = self.model.summary()

        with open(self.reports_dir / "model_summary.txt", 'w') as f:
            f.write(summary)

        metrics_dict = {}
        for k, v in self.model.metrics.items():
            task_config = self.model.task_configs.get(k)
            is_regression = task_config and task_config.task_type == "regression"
            d = v.to_dict()
            if is_regression:
                d["type"] = "regression"
                d["r2"] = d.pop("accuracy", 0)
                d.pop("f1", None)
                d.pop("auc_roc", None)
                d.pop("precision", None)
                d.pop("recall", None)
            else:
                d["type"] = "classification"
            metrics_dict[k] = d
        with open(self.reports_dir / "metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)

        importance_report = {}
        for task in self.model.models.keys():
            importance = self.model.get_feature_importance(task)
            if importance:
                importance_report[task] = dict(sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )[:15])

        with open(self.reports_dir / "feature_importance.json", 'w') as f:
            json.dump(importance_report, f, indent=2, default=str)

        return summary, metrics_dict, importance_report

    def index_for_rag(self):
        logger.info("Indexing places for RAG...")
        self.rag.index_places(self.processed_data)

    def run_full_pipeline(self):
        logger.info("=" * 60)
        logger.info("SIDEQUEST TRAINING PIPELINE - FULL REAL DATA")
        logger.info("=" * 60)

        start_time = datetime.now()

        self.load_real_data()
        self.train_model()
        summary, metrics, importance = self.evaluate_model()
        self.index_for_rag()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Duration: {duration:.2f} seconds")
        print(f"\nModel Metrics:")
        for task, metric in metrics.items():
            task_type = metric.get("type", "classification")
            if task_type == "regression":
                r2 = metric.get("r2", 0)
                rmse = metric.get("rmse", 0)
                print(f"  {task.upper()}: R²={r2:.4f}, RMSE={rmse:.4f}")
            else:
                acc = metric.get("accuracy", 0)
                f1 = metric.get("f1", 0)
                auc = metric.get("auc_roc", 0)
                print(f"  {task.upper()}: accuracy={acc:.4f}, f1={f1:.4f}, AUC-ROC={auc:.4f}")

        print(f"\nFiles saved to:")
        print(f"  Models: {self.models_dir}")
        print(f"  Reports: {self.reports_dir}")

        return {"duration": duration, "metrics": metrics, "importance": importance}


if __name__ == "__main__":
    trainer = SideQuestTrainer()
    results = trainer.run_full_pipeline()
