import streamlit as st
import pandas as pd

from core.loader import load_data
from core.features import add_features
from core.eda import plot_eda
from core.scenarios import apply_scenario
from models.ml_models import get_ml_model
from evaluation.metrics import evaluate_regression

# ===============================
# CONFIG STREAMLIT
# ===============================
st.set_page_config(layout="wide")
st.title("🚀 AI Stock Prediction Platform")

# ===============================
# UPLOAD CSV
# ===============================
file = st.file_uploader(
    "Upload market data (CSV)",
    type=["csv"]
)

model_choice = st.selectbox(
    "Choose Model",
    [
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "SVR",
        "LinearRegression"
    ]
)

scenario = st.selectbox(
    "Market Scenario",
    ["Normal", "Crash", "Bull", "Volatile"]
)

# ===============================
# MAIN PIPELINE
# ===============================
if file is not None:
    try:
        # 1. LOAD DATA
        df = load_data(file)

        st.subheader("📊 Exploratory Data Analysis")
        st.pyplot(plot_eda(df))

        # 2. SCENARIO SIMULATION
        df = apply_scenario(df, scenario)

        # 3. FEATURE ENGINEERING
        df = add_features(df)

        # 4. SPLIT X / y
        X = df.drop(columns=["close", "date"])
        y = df["close"]

        # ===============================
        # RUN MODEL
        # ===============================
        if st.button("🚀 RUN FULL PIPELINE"):
            model = get_ml_model(model_choice)
            model.fit(X, y)

            preds = model.predict(X)

            metrics = evaluate_regression(y, preds)

            st.subheader("📈 Model Performance")
            st.json(metrics)

            st.success("✅ Pipeline executed successfully!")

    except Exception as e:
        st.error("❌ Error while processing the file")
        st.exception(e)
