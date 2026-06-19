import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from feature_engineering import FeatureEngineer
from survival import CoxChurnModel
from utils import setup_logging

logger = setup_logging(Config.LOG_FILE, Config.LOG_LEVEL)

api_key_header = APIKeyHeader(name="X-API-Key")
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="SaaS Churn Survival API",
    description="Cox hazard + XGBoost churn scoring on subscription telemetry",
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

_engineer: Optional[FeatureEngineer] = None
_xgb = None
_cox: Optional[CoxChurnModel] = None
_train_df: Optional[pd.DataFrame] = None


def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    if api_key != Config.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


class PredictInput(BaseModel):
    account_age_days: int = Field(..., ge=1)
    daily_usage_mins: float = Field(..., ge=0)
    login_frequency: Literal["Daily", "Weekly", "Rarely"]


class PredictOutput(BaseModel):
    churn_probability: float
    churn_prediction: bool
    partial_hazard: float
    model_classifier: str


@app.on_event("startup")
async def startup() -> None:
    global _engineer, _xgb, _cox, _train_df
    train_path = Config.DATA_RAW / Config.TRAIN_DATA
    if train_path.exists():
        _train_df = pd.read_csv(train_path)
        _train_df.columns = _train_df.columns.str.strip()

    eng_path = Config.MODELS_DIR / Config.FEATURE_ENGINEER
    if eng_path.exists():
        _engineer = FeatureEngineer.load(eng_path)

    xgb_path = Config.MODELS_DIR / Config.MODEL_XGB
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            _xgb = pickle.load(f)

    cox_path = Config.MODELS_DIR / Config.MODEL_COX
    if cox_path.exists():
        _cox = CoxChurnModel.load(cox_path)

    logger.info("API startup complete")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "data_loaded": _train_df is not None,
        "models_loaded": _engineer is not None and _xgb is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/data/summary")
@limiter.limit("30/minute")
async def data_summary(request: Request, _: str = Depends(verify_api_key)) -> Dict[str, Any]:
    if _train_df is None:
        raise HTTPException(status_code=503, detail="Training data not loaded")
    df = _train_df
    return {
        "n_customers": len(df),
        "churn_rate": float(df[Config.TARGET_COLUMN].mean()),
        "mean_account_age_days": float(df[Config.DURATION_COLUMN].mean()),
        "mean_daily_usage_mins": float(df["Daily_Usage_Mins"].mean()),
    }


@app.get("/models/info")
async def models_info(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
    meta_path = Config.MODELS_DIR / Config.MODEL_METADATA
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {"status": "metadata not found; run train_model.py"}


@app.post("/predict", response_model=PredictOutput)
@limiter.limit("60/minute")
async def predict(
    request: Request,
    body: PredictInput,
    _: str = Depends(verify_api_key),
) -> PredictOutput:
    if _engineer is None or _xgb is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    row = pd.DataFrame(
        [
            {
                "Account_Age_Days": body.account_age_days,
                "Daily_Usage_Mins": body.daily_usage_mins,
                "Login_Frequency": body.login_frequency,
                "Churn": 0,
            }
        ]
    )
    X = _engineer.transform_X(row)
    proba = float(_xgb.predict_proba(X)[0, 1])
    pred = bool(_xgb.predict(X)[0])

    hazard = 0.0
    if _cox is not None:
        cox_cols = [c for c in _cox.covariate_columns if c in X.columns]
        hazard = float(_cox.predict_partial_hazard(X[cox_cols]).iloc[0])

    return PredictOutput(
        churn_probability=proba,
        churn_prediction=pred,
        partial_hazard=hazard,
        model_classifier="xgboost",
    )
