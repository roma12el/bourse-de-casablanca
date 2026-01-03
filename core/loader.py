import pandas as pd

def load_data(file):
    df = pd.read_csv(file)

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.lower().str.strip()

   
