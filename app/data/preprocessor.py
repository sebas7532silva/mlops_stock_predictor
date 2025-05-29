import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Convertir Datetime_ a datetime si no está
    df["Datetime_"] = pd.to_datetime(df["Datetime_"], errors='coerce')

    # Seleccionar solo columnas necesarias y eliminar nulos
    df = df[["Datetime_", "Close_AAPL"]].dropna()

    # Extraer hora y día de la semana
    df["Hour"] = df["Datetime_"].dt.hour
    df["Day"] = df["Datetime_"].dt.dayofweek

    # Devolver columnas para el entrenamiento
    return df[["Close_AAPL", "Hour", "Day"]]
