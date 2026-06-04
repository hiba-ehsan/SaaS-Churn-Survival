"""
Classification models (logistic regression, XGBoost) with standard metrics.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def classification_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}

    def train_xgboost(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict] = None
    ) -> XGBClassifier:
        params = params or {}
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        self.models["xgboost"] = model
        logger.info("XGBoost train accuracy: %.4f", model.score(X_train, y_train))
        return model

    def train_logistic(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict] = None
    ) -> LogisticRegression:
        params = params or {}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        self.models["logistic"] = model
        logger.info("Logistic train accuracy: %.4f", model.score(X_train, y_train))
        return model

    def evaluate(self, model, X_test, y_test, name: str) -> Dict[str, float]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, y_pred, y_proba)
        logger.info("%s metrics: %s", name, {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    def compare(self, X_test, y_test) -> pd.DataFrame:
        rows = []
        for name, model in self.models.items():
            m = self.evaluate(model, X_test, y_test, name)
            rows.append({"model": name, **m})
        return pd.DataFrame(rows)

    @staticmethod
    def save_model(model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load_model(path: Path) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_metadata(path: Path, metadata: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
