# SaaS Churn Survival Prediction Pipeline

**Advanced churn modeling using Survival Analysis + Machine Learning for subscription platforms**

An end-to-end quantitative analytics pipeline that combines **Cox Proportional Hazards** (Survival Analysis) with **XGBoost** classification to predict customer churn and model time-to-event in SaaS/subscription businesses.

### Project Context
This project was developed as an extension of my internship work at **Thingsty** (an IoT facility management platform). To respect data confidentiality, the pipeline was built and demonstrated using a public **SaaS Customer Churn Prediction Dataset** from Kaggle as a representative proxy for real subscription telemetry data.

→ Dataset: [SaaS Customer Churn Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/suhanigupta04/saas-customer-churn-prediction-dataset)

### The Quantitative Angle
While most churn models focus only on binary classification, this pipeline models **hazard rates over time** using survival analysis — the same approach used by growth investors and quant teams to evaluate SaaS companies and estimate Lifetime Value (LTV).

### Key Features

- **Survival Analysis**: Cox Proportional Hazards model to estimate churn hazard rates and understand time-to-churn.
- **ML Classification**: XGBoost + Logistic Regression baseline for churn probability scoring.
- **Automated Feature Engineering**: Account age, usage intensity, login frequency, and behavioral metrics.
- **Production Components**:
  - FastAPI backend with rate limiting and API key authentication
  - Interactive Streamlit dashboard for exploration and real-time predictions
- **Evaluation**: Concordance Index, F1-score, ROC-AUC, feature importance, and hazard ratio interpretation.

### Tech Stack

- **Languages & Core**: Python, Pandas, NumPy
- **Survival Analysis**: `lifelines` (Cox PH)
- **ML**: XGBoost, scikit-learn
- **Backend**: FastAPI + SlowAPI
- **Dashboard**: Streamlit
- **Others**: Matplotlib/Seaborn, Pickle

### Quick Start

```bash
git clone https://github.com/yourusername/saas-churn-survival.git
cd saas-churn-survival

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Train models
python train_model.py

# Launch Dashboard
streamlit run visual_analysis.py

# Launch API
uvicorn api:app --reload --port 8000