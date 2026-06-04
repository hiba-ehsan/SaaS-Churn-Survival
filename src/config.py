"""
Pipeline configuration — SaaS churn / survival analysis.
"""

import os
from pathlib import Path
from typing import Any, Dict

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent

    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"

    for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    TRAIN_DATA = "train.csv"
    TEST_DATA = "test_.csv"

    TARGET_COLUMN = "Churn"
    DURATION_COLUMN = "Account_Age_Days"  # time at risk (days)
    EVENT_COLUMN = "Churn"  # 1 = churn event, 0 = right-censored

    # Do not use these as Cox covariates when duration is Account_Age_Days (leakage)
    COX_EXCLUDE_COVARIATES = frozenset(
        {"Account_Age_Days", "Account_Age_Log", "Account_Age_Sqrt"}
    )

    ID_COLUMNS = ("Customer_ID", "Name", "Email", "Last_Support_Ticket")
    NUMERIC_FEATURES = ("Account_Age_Days", "Daily_Usage_Mins")
    CATEGORICAL_FEATURES = ("Login_Frequency",)

    MODEL_XGB = "churn_xgboost.pkl"
    MODEL_LR = "churn_logistic.pkl"
    MODEL_COX = "cox_ph.pkl"
    FEATURE_ENGINEER = "feature_engineer.pkl"
    MODEL_METADATA = "model_metadata.json"

    API_KEY = os.getenv("API_KEY", "dev-api-key")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    ML_PARAMS: Dict[str, Any] = {
        "random_state": 42,
        "test_size": 0.2,
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "random_state": 42,
        },
        "logistic_regression": {
            "max_iter": 2000,
            "random_state": 42,
            "class_weight": "balanced",
        },
    }

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = PROJECT_ROOT / "logs" / "pipeline.log"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
