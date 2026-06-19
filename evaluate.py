import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from models import ModelTrainer
from survival import CoxChurnModel
from utils import setup_logging
from lifelines.utils import concordance_index

logger = setup_logging(Config.LOG_FILE, Config.LOG_LEVEL)


def main() -> None:
    engineer = FeatureEngineer.load(Config.MODELS_DIR / Config.FEATURE_ENGINEER)
    test_df = DataLoader(Config.DATA_RAW).load_csv(Config.TEST_DATA)
    test_df = DataLoader(Config.DATA_RAW).clean_data(test_df)

    X_test = engineer.transform_X(test_df)
    y_test = test_df[Config.TARGET_COLUMN]

    trainer = ModelTrainer()
    trainer.models["xgboost"] = ModelTrainer.load_model(Config.MODELS_DIR / Config.MODEL_XGB)
    trainer.models["logistic"] = ModelTrainer.load_model(Config.MODELS_DIR / Config.MODEL_LR)

    comparison = trainer.compare(X_test, y_test)
    print(comparison.to_string(index=False))

    cox = CoxChurnModel.load(Config.MODELS_DIR / Config.MODEL_COX)
    prepared = engineer.prepare_features(test_df, fit=False)
    prepared[Config.DURATION_COLUMN] = test_df[Config.DURATION_COLUMN].values
    cox_covariates = [
        c for c in engineer.feature_names if c not in Config.COX_EXCLUDE_COVARIATES
    ]
    partial = cox.predict_partial_hazard(prepared[cox_covariates])
    ci = concordance_index(
        prepared[Config.DURATION_COLUMN],
        -partial,
        prepared[Config.EVENT_COLUMN],
    )
    print(f"\nCox concordance (test): {ci:.4f}")
    print("\nCox coefficients:\n", cox.summary_frame())


if __name__ == "__main__":
    main()
