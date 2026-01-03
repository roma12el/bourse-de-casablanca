import pandas as pd

def load_data(file):
    df = pd.read_csv(file)

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.lower().str.strip()

    # Détection automatique de la colonne date
    date_candidates = ['date', 'datetime', 'time', 'timestamp']

    date_col = None
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            f"Aucune colonne date trouvée. Colonnes disponibles : {df.columns.tolist()}"
        )

    df[date_col] = pd.to_datetime(df[date_col])

    # Renommer en 'date' pour standardisation
    df.rename(columns={date_col: 'date'}, inplace=True)

    return df.sort_values('date')
