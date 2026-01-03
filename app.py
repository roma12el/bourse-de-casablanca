
import streamlit as st
import pandas as pd
from core.loader import load_data
from core.features import add_features
from core.eda import plot_eda
from core.scenarios import apply_scenario
from models.ml_models import get_ml_model
from models.dl_models import lstm_model, gru_model
from evaluation.metrics import evaluate_regression

st.set_page_config(layout="wide")
st.title("🚀 AI Stock Prediction Platform")

file = st.file_uploader("Upload Excel", type=["xlsx"])
model_choice = st.selectbox("Choose Model", ["RandomForest","XGBoost","LightGBM","SVR","LinearRegression","LSTM","GRU"])
scenario = st.selectbox("Market Scenario", ["Normal","Crash","Bull","Volatile"])

if file:
    df = load_data(file)
    st.pyplot(plot_eda(df))
    df = apply_scenario(df, scenario)
    df = add_features(df)

    X = df.drop(columns=["close","date"])
    y = df["close"]

    if st.button("RUN FULL PIPELINE"):
        if model_choice in ["LSTM","GRU"]:
            st.success("Deep Learning model trained (demo structure).")
        else:
            model = get_ml_model(model_choice)
            model.fit(X, y)
            preds = model.predict(X)
            metrics = evaluate_regression(y, preds)
            st.json(metrics)
