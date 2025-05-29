import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from app.utils.config import MODELS_PATH

def evaluate_and_promote(new_model, X, y, threshold_rmse=1.5):
    y_pred = new_model.predict(X)
    new_rmse = mean_squared_error(y, y_pred)

    # Cargar modelo anterior si existe
    try:
        old_model = load(f"{MODELS_PATH}/latest_model.joblib")
        old_rmse = mean_squared_error(y, old_model.predict(X))
    except:
        old_rmse = float("inf")

    if new_rmse < old_rmse:
        dump(new_model, f"{MODELS_PATH}/latest_model.joblib")
        print(f"✅ Modelo promovido con RMSE: {new_rmse:.2f}")
    else:
        print(f"⚠️ Modelo NO promovido. RMSE anterior: {old_rmse:.2f}, nuevo: {new_rmse:.2f}")
