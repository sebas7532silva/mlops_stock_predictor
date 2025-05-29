import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
import os

def fetch_hourly_data(ticker="AAPL", hours=15):
    data = yf.download(ticker, period="7d", interval="1h", auto_adjust=False)
    data.reset_index(inplace=True)
    return data.tail(hours)

def save_to_mongo(df, db_name="stock_db", collection_name="aapl"):
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client[db_name]
    collection = db[collection_name]

    # Eliminar duplicados (opcional)
    collection.delete_many({})

    df = df.copy()
    df["Datetime"] = df["Datetime"].astype(str)

    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    collection.insert_many(df.to_dict("records"))

    print(f"{len(df)} registros guardados en MongoDB.")

def load_data_from_mongo(db_name="stock_db", collection_name="aapl"):
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client[db_name]
    collection = db[collection_name]

    data = list(collection.find({}))
    if not data:
        print("No se encontraron datos en MongoDB.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Si tienes la columna 'Datetime' en formato string, la convierto a datetime
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])

    return df

if __name__ == "__main__":
    df = fetch_hourly_data()
    save_to_mongo(df)
