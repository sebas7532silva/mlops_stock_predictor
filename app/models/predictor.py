import os
import pandas as pd
from datetime import datetime
from joblib import load
from pymongo import MongoClient
from app.utils.config import MODELS_PATH

def load_latest_model():
    return load(f"{MODELS_PATH}/latest_model.joblib")

def predict_next_price(model):
    now = datetime.now()
    features = pd.DataFrame([[now.hour, now.weekday()]], columns=["Hour", "Day"])
    return model.predict(features)[0]

def load_latest_data(db_name="stock_db", collection_name="aapl", limit=100):
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client[db_name]
    collection = db[collection_name]

    cursor = collection.find({}, {"_id": 0, "Datetime_": 1, "Close_AAPL": 1}).sort("Datetime_", -1).limit(limit)
    data = list(cursor)

    df = pd.DataFrame(data)
    df["Datetime_"] = pd.to_datetime(df["Datetime_"])
    df = df.sort_values("Datetime_").reset_index(drop=True)
    return df

