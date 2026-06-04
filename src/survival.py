"""
Cox proportional hazards — churn as survival event, account age as duration.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

logger = logging.getLogger(__name__)


class CoxChurnModel:
    """Semi-parametric hazard model: h(t|X) = h0(t) * exp(beta'X)."""

    def __init__(self):
        self.cph = CoxPHFitter(penalizer=0.01)
        self.covariate_columns: List[str] = []
        self.metrics: Dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        covariate_cols: List[str],
    ) -> "CoxChurnModel":
        use_cols = [duration_col, event_col] + covariate_cols
        data = df[use_cols].dropna().copy()
        data = data[data[duration_col] > 0]

        self.covariate_columns = covariate_cols
        self.cph.fit(data, duration_col=duration_col, event_col=event_col)

        partial_hazard = self.cph.predict_partial_hazard(data)
        self.metrics["concordance_index"] = float(
            concordance_index(
                data[duration_col],
                -partial_hazard,
                data[event_col],
            )
        )
        logger.info("Cox PH concordance index: %.4f", self.metrics["concordance_index"])
        return self

    def summary_frame(self) -> pd.DataFrame:
        return self.cph.summary

    def predict_partial_hazard(self, X: pd.DataFrame) -> pd.Series:
        aligned = X[self.covariate_columns].copy()
        return self.cph.predict_partial_hazard(aligned)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "CoxChurnModel":
        with open(path, "rb") as f:
            return pickle.load(f)
