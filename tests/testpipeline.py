import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import ModelTrainer


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "Account_Age_Days": [100, 200, 300, 400, 500],
            "Daily_Usage_Mins": [10, 20, 30, 40, 50],
            "Login_Frequency": ["Daily", "Weekly", "Rarely", "Daily", "Weekly"],
            "Churn": [1, 0, 0, 1, 0],
        }
    )


def test_config_paths():
    assert Config.DATA_RAW.exists()
    assert Config.MODELS_DIR.exists()


def test_feature_engineer(sample_data):
    engineer = FeatureEngineer()
    prepared = engineer.prepare_features(sample_data, fit=True)
    assert Config.TARGET_COLUMN in prepared.columns
    assert len(engineer.feature_names) >= 3


def test_xgboost_train(sample_data):
    engineer = FeatureEngineer()
    prepared = engineer.prepare_features(sample_data, fit=True)
    X = prepared.drop(columns=[Config.TARGET_COLUMN])
    y = prepared[Config.TARGET_COLUMN]
    trainer = ModelTrainer()
    model = trainer.train_xgboost(X, y, {"n_estimators": 10, "max_depth": 2, "random_state": 42})
    assert len(model.predict(X)) == len(X)


def test_real_data_loads():
    if (Config.DATA_RAW / Config.TRAIN_DATA).exists():
        loader = DataLoader(Config.DATA_RAW)
        df = loader.load_csv(Config.TRAIN_DATA)
        assert len(df) > 100
        assert Config.TARGET_COLUMN in df.columns
