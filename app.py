import pandas as pd

def load_data(file):
    # Détection automatique du type de fichier
    filename = file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file, sep=None, engine="python")
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format")

    # Nettoyage des colonnes
    df.columns = df.columns.str.lower().str.strip()

    # Détection intelligente de la colonne date
    date_candidates = ["date", "datetime", "time", "timestamp"]

    date_col = None
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            f"No date column found. Available columns: {df.columns.tolist()}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df.rename(columns={date_col: "date"}, inplace=True)

    return df.sort_values("date")
