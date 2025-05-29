import os
from joblib import dump
from app.data.fetcher import load_data_from_mongo
from app.data.preprocessor import preprocess_data
from app.models.trainer import train_model
from app.utils.config import MODELS_PATH
from app.models.trainer import send_telegram_message


def main():
    try:
        os.makedirs(MODELS_PATH, exist_ok=True)

        df = load_data_from_mongo()
        df = preprocess_data(df)

        model = train_model(df)

        model_path = os.path.join(MODELS_PATH, "latest_model.joblib")
        dump(model, model_path)
        

        print(f"Modelo entrenado y guardado en {model_path}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
    
    send_telegram_message("Entrenamiento completado con éxito")

if __name__ == "__main__":
    main()
    

