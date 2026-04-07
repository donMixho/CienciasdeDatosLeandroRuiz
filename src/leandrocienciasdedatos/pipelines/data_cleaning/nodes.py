import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Aquí va tu lógica del notebook
    df = df.drop_duplicates()
    df = df.fillna(0) # O la lógica que usamos
    return df