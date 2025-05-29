import os
from joblib import dump
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import mlflow
import mlflow.sklearn
from pymongo import MongoClient
from app.utils.config import MODELS_PATH
from app.data.preprocessor import preprocess_data 
import matplotlib.pyplot as plt
from datetime import datetime

import requests

def send_telegram_message(message):
    token = "8008337527:AAE8HIMXKR8MHhx3a_5XGE9HMmx0IjKXYmY"
    chat_id = 5607093141
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Error al enviar mensaje Telegram:", e)

def plot_pred_vs_actual(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='45° Line')
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def load_data_from_mongo(db_name="stock_db", collection_name="aapl"):
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find({}))
    df = pd.DataFrame(data)
    return df

def train_model(df):
    X = df[["Hour", "Day"]]
    y = df["Close_AAPL"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 7,  9],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    plot_pred_vs_actual(y_train, y_train_pred, "Train: Predicción vs Valor real", "train_pred_vs_actual.png")
    plot_pred_vs_actual(y_val, y_val_pred, "Validación: Predicción vs Valor real", "val_pred_vs_actual.png")
    
    rmse_train = mean_squared_error(y_train, y_train_pred)
    rmse_val = mean_squared_error(y_val, y_val_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)

    EXPERIMENT_NAME = "AAPL_Prediction"
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_val", rmse_val)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_val", r2_val)
        mlflow.log_params(best_params)
        mlflow.log_artifact("train_pred_vs_actual.png")
        mlflow.log_artifact("val_pred_vs_actual.png")
    
    send_telegram_message(
    f"✅ Entrenamiento completado a las {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"RMSE val: {rmse_val:.4f} | R2 val: {r2_val:.4f}"
        )

    return best_model

if __name__ == "__main__":
    os.makedirs(MODELS_PATH, exist_ok=True)

    # Cargar datos desde MongoDB
    df = load_data_from_mongo()

    # Preprocesar datos para agregar Hour y Day, eliminar nulos, etc.
    df = preprocess_data(df)

    # Entrenar modelo
    model = train_model(df)

    # Guardar modelo localmente
    dump(model, os.path.join(MODELS_PATH, "latest_model.joblib"))

    print(f"Modelo entrenado y guardado en {MODELS_PATH}/latest_model.joblib")



