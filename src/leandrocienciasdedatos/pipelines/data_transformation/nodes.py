import pandas as pd

def integrate_data(emp, eva, cap, aus) -> pd.DataFrame:
    # Aquí pegas la lógica del Pipeline 3 de tu notebook
    # El merge que nos dio los 255 registros
    df_final = emp.merge(eva, on="id_empleado", how="left")
    # ... (resto de tus merges)
    return df_final