import os

# Carpeta donde se guarda el modelo
MODELS_PATH = os.path.join(os.getcwd(), "app", "models", "artifacts")
os.makedirs(MODELS_PATH, exist_ok=True)
