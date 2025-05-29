import pandas as pd

def preprocess_data(df, n_lags=3, rolling_window=3):
    df = df.copy()

    # Convertir Datetime_ a datetime
    df["Datetime_"] = pd.to_datetime(df["Datetime_"], errors='coerce')
    df = df[["Datetime_", "Close_AAPL"]].dropna()

    # Ordenar por fecha por si acaso
    df = df.sort_values("Datetime_")

    # Extraer características de tiempo
    df["Hour"] = df["Datetime_"].dt.hour
    df["DayOfWeek"] = df["Datetime_"].dt.dayofweek

    # Crear columnas de lags
    for lag in range(1, n_lags + 1):
        df[f"Close_lag_{lag}"] = df["Close_AAPL"].shift(lag)

    # Crear medias móviles
    df[f"rolling_mean_{rolling_window}"] = df["Close_AAPL"].rolling(window=rolling_window).mean()
    df[f"rolling_std_{rolling_window}"] = df["Close_AAPL"].rolling(window=rolling_window).std()

    # Eliminar valores nulos que se crean por los lags y rolling
    df = df.dropna()

    # Variables para entrenamiento (X) y target (y)
    feature_cols = [col for col in df.columns if col.startswith("Close_lag_") or 
                    col.startswith("rolling_") or col in ["Hour", "DayOfWeek"]]


    return df, feature_cols

