import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.models.predictor import load_latest_model, predict_next_price, load_latest_data

def test_load_model():
    model = load_latest_model()
    assert model is not None
    # Puedes agregar aquí chequear tipo, atributos, etc.

def test_load_data():
    df = load_latest_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Datetime_" in df.columns
    assert "Close_AAPL" in df.columns

def test_predict_next_price():
    model = load_latest_model()
    pred = predict_next_price(model)
    assert isinstance(pred, float) or isinstance(pred, int)
    assert pred > 0  # Precios deben ser positivos

