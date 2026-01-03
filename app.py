import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide")
st.title("🚀 AI Stock Prediction Platform (SAFE MODE)")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower().str.strip()

        st.write("Columns detected:", df.columns.tolist())

       # Nettoyage des noms de colonnes
df.columns = df.columns.str.lower().str.strip()

# Renommer la colonne date si nécessaire
if "exchange date" in df.columns:
    df.rename(columns={"exchange date": "date"}, inplace=True)

# Vérification finale
if "date" not in df.columns:
    st.error(f"No date column found. Columns detected: {df.columns.tolist()}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")


        st.line_chart(df.set_index("date")["close"])

        df["lag1"] = df["close"].shift(1)
        df.dropna(inplace=True)

        X = df[["lag1"]]
        y = df["close"]

        if st.button("RUN MODEL"):
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
            preds = model.predict(X)

            rmse = mean_squared_error(y, preds, squared=False)
            st.success(f"✅ Model ran successfully | RMSE = {rmse:.4f}")

    except Exception as e:
        st.error("❌ ERROR")
        st.exception(e)

