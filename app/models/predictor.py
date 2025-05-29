import os
import pandas as pd
from datetime import datetime
from joblib import load
from pymongo import MongoClient
from app.utils.config import MODELS_PATH
from mlflow.tracking import MlflowClient

def load_latest_model():
    return load(f"{MODELS_PATH}/latest_model.joblib")

def load_latest_features():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("AAPL_Prediction")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    latest_run = runs[0] 
    path = client.download_artifacts(latest_run.info.run_id, "latest_features.csv")
    return pd.read_csv(path)

def predict_next_price(model):
    features = load_latest_features()
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

