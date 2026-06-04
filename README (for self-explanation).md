# SaaS Churn Survival Pipeline

Quantitative churn modeling for subscription SaaS: **Cox proportional hazards** (semi-parametric survival) plus **XGBoost** classification on account telemetry. Built for portfolio use as survival/LTV-style analysis, not generic “IoT startup” demos.


## Problem setup

| Symbol | Column | Role |
|--------|--------|------|
| \(T\) | `Account_Age_Days` | Time at risk (days since signup) |
| \(\delta\) | `Churn` | Event indicator (1 = churned, 0 = censored / still active) |
| \(X\) | Usage + login features | Covariates in Cox PH and classifiers |

**Cox model:** \(h(t \mid X) = h_0(t)\,\exp(\beta^\top X)\).  
**Classifier:** \(P(\delta=1 \mid X)\) via XGBoost (logistic regression as interpretable baseline).

## Data

- `data/raw/train.csv` — 2,000 accounts  
- `data/raw/test_.csv` — 500 accounts  
- Columns: `Account_Age_Days`, `Daily_Usage_Mins`, `Login_Frequency`, `Churn` (+ IDs dropped at train time)

## Quick start

```powershell
cd "c:\Codes\User Analysis\thingsty-saas-analytics"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python evaluate_model.py
pytest tests/ -q
```

**Dashboard**

```powershell
streamlit run visual_analysis.py
```

**API**

```powershell
$env:API_KEY = "dev-api-key"
uvicorn api:app --reload --port 8000
```

```powershell
curl -H "X-API-Key: dev-api-key" http://localhost:8000/health
curl -H "X-API-Key: dev-api-key" -H "Content-Type: application/json" `
  -d '{"account_age_days":365,"daily_usage_mins":30,"login_frequency":"Weekly"}' `
  http://localhost:8000/predict
```

## Layout

```
src/
  config.py              # paths, column roles
  data_loader.py         # CSV load + clean
  feature_engineering.py # covariates (fit/transform + pickle)
  survival.py            # Cox PH (lifelines)
  models.py              # XGBoost + logistic
train_model.py           # full training pipeline
evaluate_model.py        # holdout metrics + Cox summary
api.py                   # FastAPI scoring
visual_analysis.py       # Streamlit lab
models/                  # artifacts after training
```

## Deploy to GitHub

From the project folder (init repo if needed):

```powershell
git init
git add .
git commit -m "SaaS churn survival pipeline: Cox PH + XGBoost on subscription telemetry"
git branch -M main
git remote add origin https://github.com/YOUR_USER/saas-churn-survival.git
git push -u origin main
```

Replace `YOUR_USER/saas-churn-survival` with your repository. Do not commit `.env` or virtualenv folders.

## License

MIT
