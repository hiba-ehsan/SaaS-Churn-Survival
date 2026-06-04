"""
Feature engineering for SaaS subscription churn (classification + survival covariates).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Fit on train; transform train/test with identical columns."""

    def __init__(self, target_column: str = Config.TARGET_COLUMN):
        self.target_column = target_column
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._dummy_columns: List[str] = []

    def identify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        categorical = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c != self.target_column
        ]
        numerical = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != self.target_column
        ]
        self.categorical_features = categorical
        self.numerical_features = numerical
        return categorical, numerical

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "Account_Age_Days" in out.columns:
            out["Account_Age_Log"] = np.log1p(out["Account_Age_Days"])
            out["Account_Age_Sqrt"] = np.sqrt(out["Account_Age_Days"].clip(lower=0))

        if "Daily_Usage_Mins" in out.columns and "Account_Age_Days" in out.columns:
            # Usage intensity: minutes per day of account life (engagement hazard covariate)
            out["Usage_Per_Day"] = out["Daily_Usage_Mins"] / (out["Account_Age_Days"] + 1.0)
            out["Usage_Log"] = np.log1p(out["Daily_Usage_Mins"].clip(lower=0))

        if "Login_Frequency" in out.columns:
            freq_map = {"Daily": 3, "Weekly": 2, "Rarely": 1}
            out["Login_Frequency_Ordinal"] = (
                out["Login_Frequency"].map(freq_map).fillna(0).astype(float)
            )

        return out

    def _drop_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in df.columns if c in Config.ID_COLUMNS and c != self.target_column]
        if drop:
            df = df.drop(columns=drop)
            logger.info("Dropped ID/text columns: %s", drop)
        return df

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if not self.categorical_features:
            return df

        out = df.copy()
        for col in self.categorical_features:
            if col not in out.columns:
                continue
            dummies = pd.get_dummies(out[col], prefix=col, drop_first=True, dtype=float)
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)

        if fit:
            self._dummy_columns = [c for c in out.columns if c != self.target_column]
        elif self._dummy_columns:
            for col in self._dummy_columns:
                if col not in out.columns:
                    out[col] = 0.0
            ordered = ([self.target_column] if self.target_column in out.columns else []) + self._dummy_columns
            out = out[ordered]

        return out

    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        cols = [c for c in self.numerical_features if c in df.columns]
        if not cols:
            return df

        out = df.copy()
        if fit:
            out[cols] = self.scaler.fit_transform(out[cols])
        else:
            out[cols] = self.scaler.transform(out[cols])
        return out

    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = False,
        encode: bool = True,
        scale: bool = True,
        engineer: bool = True,
    ) -> pd.DataFrame:
        out = df.copy()

        for col in out.columns:
            if out[col].isnull().any():
                if pd.api.types.is_numeric_dtype(out[col]):
                    out[col] = out[col].fillna(out[col].median())
                else:
                    out[col] = out[col].fillna("Unknown")

        out = self._drop_ids(out)

        if engineer:
            out = self.engineer_features(out)

        if fit or not self.categorical_features:
            self.identify_features(out)

        if encode:
            out = self.encode_categorical_features(out, fit=fit)

        if scale:
            out = self.scale_numerical_features(out, fit=fit)

        num_cols = out.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if out[col].isnull().any():
                out[col] = out[col].fillna(out[col].median())

        self.feature_names = [c for c in out.columns if c != self.target_column]
        return out

    def transform_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature matrix aligned to training columns."""
        prepared = self.prepare_features(df, fit=False)
        X = prepared.drop(columns=[self.target_column], errors="ignore")
        missing = set(self.feature_names) - set(X.columns)
        for col in missing:
            X[col] = 0.0
        return X[self.feature_names]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved feature engineer to %s", path)

    @staticmethod
    def load(path: Path) -> "FeatureEngineer":
        with open(path, "rb") as f:
            return pickle.load(f)
