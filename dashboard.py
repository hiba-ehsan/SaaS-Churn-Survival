import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from data_loader import DataLoader

st.set_page_config(page_title="SaaS Churn Survival Lab", layout="wide")
st.title("SaaS Churn Survival Lab")
st.caption("Cox proportional hazards & gradient boosting on subscription telemetry (2,500 accounts)")

loader = DataLoader(Config.DATA_RAW)
if not (Config.DATA_RAW / Config.TRAIN_DATA).exists():
    st.error("Place train.csv in data/raw/")
    st.stop()

df = loader.load_csv(Config.TRAIN_DATA)
df = loader.clean_data(df)

col1, col2, col3, col4 = st.columns(4)
churn_rate = df[Config.TARGET_COLUMN].mean()
col1.metric("Accounts", f"{len(df):,}")
col2.metric("Churn rate", f"{churn_rate:.1%}")
col3.metric("Mean time at risk (days)", f"{df[Config.DURATION_COLUMN].mean():.0f}")
col4.metric("Mean usage (min/day proxy)", f"{df['Daily_Usage_Mins'].mean():.1f}")

tab1, tab2, tab3 = st.tabs(["Hazard view", "Predict", "Model metrics"])

with tab1:
    st.subheader("Churn rate vs account age (time at risk)")
    df["age_bin"] = pd.cut(df[Config.DURATION_COLUMN], bins=8)
    hazard_table = (
        df.groupby("age_bin", observed=True)[Config.TARGET_COLUMN]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
    )
    st.dataframe(hazard_table, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    hazard_table["churn_rate"].plot(kind="bar", ax=ax, color="#c0392b")
    ax.set_ylabel("Empirical churn rate")
    ax.set_xlabel("Account age bin (days)")
    st.pyplot(fig)

    st.subheader("Churn by login frequency")
    freq = df.groupby("Login_Frequency")[Config.TARGET_COLUMN].mean()
    st.bar_chart(freq)

with tab2:
    st.subheader("Classifier + partial hazard")
    age = st.number_input("Account age (days)", 1, 2000, 365)
    usage = st.number_input("Daily usage (minutes)", 0.0, 500.0, 30.0)
    login = st.selectbox("Login frequency", ["Daily", "Weekly", "Rarely"])

    if st.button("Score account"):
        try:
            import pickle

            engineer_path = Config.MODELS_DIR / Config.FEATURE_ENGINEER
            xgb_path = Config.MODELS_DIR / Config.MODEL_XGB
            if not engineer_path.exists() or not xgb_path.exists():
                st.warning("Run: python train_model.py")
                st.stop()

            from feature_engineering import FeatureEngineer

            engineer = FeatureEngineer.load(engineer_path)
            with open(xgb_path, "rb") as f:
                xgb = pickle.load(f)

            row = pd.DataFrame(
                [
                    {
                        "Account_Age_Days": age,
                        "Daily_Usage_Mins": usage,
                        "Login_Frequency": login,
                        "Churn": 0,
                    }
                ]
            )
            X = engineer.transform_X(row)
            p = float(xgb.predict_proba(X)[0, 1])
            st.metric("P(churn)", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
        except Exception as exc:
            st.error(str(exc))

with tab3:
    meta_path = Config.MODELS_DIR / Config.MODEL_METADATA
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            st.json(json.load(f))
    else:
        st.info("Train models first: python train.py")
