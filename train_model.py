#!/usr/bin/env python
"""
Train SaaS churn models: Cox proportional hazards + XGBoost + logistic baseline.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import ModelTrainer
from survival import CoxChurnModel
from utils import get_data_summary, setup_logging

logger = setup_logging(Config.LOG_FILE, Config.LOG_LEVEL)


def main() -> None:
    logger.info("=" * 60)
    logger.info("SaaS Churn / Survival Training Pipeline")
    logger.info("=" * 60)

    train_path = Config.DATA_RAW / Config.TRAIN_DATA
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}")

    train_df, test_df = DataLoader.load_train_test_split(
        Config.DATA_RAW, Config.TRAIN_DATA, Config.TEST_DATA, clean=True
    )
    logger.info("Train %s | Test %s", train_df.shape, test_df.shape)
    logger.info("Summary: %s", get_data_summary(train_df))

    engineer = FeatureEngineer(target_column=Config.TARGET_COLUMN)
    train_prepared = engineer.prepare_features(train_df, fit=True)
    test_prepared = engineer.prepare_features(test_df, fit=False)

    X_train = engineer.transform_X(train_df)
    y_train = train_prepared[Config.TARGET_COLUMN]
    X_test = engineer.transform_X(test_df)
    y_test = test_prepared[Config.TARGET_COLUMN]

    trainer = ModelTrainer(random_state=Config.ML_PARAMS["random_state"])

    lr = trainer.train_logistic(X_train, y_train, Config.ML_PARAMS["logistic_regression"])
    xgb = trainer.train_xgboost(X_train, y_train, Config.ML_PARAMS["xgboost"])

    lr_metrics = trainer.evaluate(lr, X_test, y_test, "logistic")
    xgb_metrics = trainer.evaluate(xgb, X_test, y_test, "xgboost")
    comparison = trainer.compare(X_test, y_test)
    logger.info("\n%s", comparison.to_string(index=False))

    f1_gain_pct = 0.0
    if lr_metrics["f1"] > 0:
        f1_gain_pct = (xgb_metrics["f1"] - lr_metrics["f1"]) / lr_metrics["f1"] * 100.0

    # Cox PH: duration = raw account age; covariates exclude duration (time axis vs X)
    cox_covariates = [
        c for c in engineer.feature_names if c not in Config.COX_EXCLUDE_COVARIATES
    ]

    cox_train = train_prepared[cox_covariates + [Config.EVENT_COLUMN]].copy()
    cox_train[Config.DURATION_COLUMN] = train_df[Config.DURATION_COLUMN].values
    cox_model = CoxChurnModel().fit(
        cox_train,
        duration_col=Config.DURATION_COLUMN,
        event_col=Config.EVENT_COLUMN,
        covariate_cols=cox_covariates,
    )

    cox_test = test_prepared[cox_covariates + [Config.EVENT_COLUMN]].copy()
    cox_test[Config.DURATION_COLUMN] = test_df[Config.DURATION_COLUMN].values
    partial = cox_model.predict_partial_hazard(cox_test[cox_covariates])
    from lifelines.utils import concordance_index

    test_ci = float(
        concordance_index(
            cox_test[Config.DURATION_COLUMN],
            -partial,
            cox_test[Config.EVENT_COLUMN],
        )
    )

    best_row = comparison.sort_values("f1", ascending=False).iloc[0]
    best_model = best_row["model"]

    ModelTrainer.save_model(xgb, Config.MODELS_DIR / Config.MODEL_XGB)
    ModelTrainer.save_model(lr, Config.MODELS_DIR / Config.MODEL_LR)
    cox_model.save(Config.MODELS_DIR / Config.MODEL_COX)
    engineer.save(Config.MODELS_DIR / Config.FEATURE_ENGINEER)

    metadata = {
        "project": "saas-churn-survival",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "churn_rate_train": float(train_df[Config.TARGET_COLUMN].mean()),
        "feature_columns": engineer.feature_names,
        "cox_covariates": cox_covariates,
        "duration_column": Config.DURATION_COLUMN,
        "event_column": Config.EVENT_COLUMN,
        "metrics_test": {
            "logistic": lr_metrics,
            "xgboost": xgb_metrics,
            "cox_concordance_train": cox_model.metrics["concordance_index"],
            "cox_concordance_test": test_ci,
            "f1_improvement_xgb_vs_logistic_pct": round(f1_gain_pct, 2),
        },
        "best_classifier": best_model,
        "training_date": pd.Timestamp.now().isoformat(),
    }
    ModelTrainer.save_metadata(Config.MODELS_DIR / Config.MODEL_METADATA, metadata)

    logger.info("Done. Models saved to %s", Config.MODELS_DIR)
    logger.info("XGBoost F1 lift vs logistic: %.2f%%", f1_gain_pct)


if __name__ == "__main__":
    main()
