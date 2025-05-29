import streamlit as st
import pandas as pd
import altair as alt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.models.predictor import load_latest_model, predict_next_price, load_latest_data
from ml_flow_front import get_last_metrics, get_model_parameters, get_model_artifacts

st.title("📈 Predicción de AAPL")

st.subheader("Última predicción de precios de AAPL")

# Cargar modelo
model = load_latest_model()

metrics = get_last_metrics("AAPL_Prediction")

params = get_model_parameters("AAPL_Prediction")
artifact_images = get_model_artifacts("AAPL_Prediction")
    
df = load_latest_data()
df = df.sort_values("Datetime_").tail(5).copy()

# Predecir el siguiente precio
next_price = predict_next_price(model)

# Crear fila para la predicción
last_datetime = df["Datetime_"].iloc[-1]
# Suponemos que el intervalo es 1 hora para el siguiente punto
next_datetime = last_datetime + pd.Timedelta(hours=1)
pred_row = pd.DataFrame({
    "Datetime_": [next_datetime],
    "Close_AAPL": [next_price]
})

# Concatenar con datos históricos para graficar
df_all = pd.concat([df, pred_row], ignore_index=True)

# Para hacer líneas con colores distintos, creamos una columna para distinguir segmentos
df_all["segment"] = "histórico"
df_all.loc[df_all.index[-2:], "segment"] = ["histórico", "predicción"]

import altair as alt

# Puntos para toda la serie (histórico + predicción), usando el color del segmento
points_all = alt.Chart(df_all).mark_point(filled=True, size=80).encode(
    x='Datetime_:T',
    y='Close_AAPL:Q',
    color=alt.Color('segment:N',
                    scale=alt.Scale(domain=["histórico", "predicción"],
                                    range=["blue", "red"]),
                    legend=None)
)

# Texto con valores sobre cada punto, ligeramente arriba para que no se solapen con el punto
text_labels = alt.Chart(df_all).mark_text(
    align='center',
    baseline='bottom',
    dy=-10,  # desplazamiento vertical hacia arriba
    fontSize=12,
    fontWeight='bold'
).encode(
    x='Datetime_:T',
    y='Close_AAPL:Q',
    color=alt.Color('segment:N',
                    scale=alt.Scale(domain=["histórico", "predicción"],
                                    range=["blue", "red"]),
                    legend=None),
    text=alt.Text('Close_AAPL:Q', format=".2f")
)

line_chart = alt.Chart(df_all).mark_line().encode(
    x=alt.X('Datetime_:T', title='Fecha y Hora'),
    y=alt.Y('Close_AAPL:Q', title='Precio'),
    color=alt.Color('segment:N',
                    scale=alt.Scale(domain=["histórico", "predicción"],
                                    range=["blue", "red"]),
                    legend=None)
).properties(
    width=700,
    height=400
)

# Combina todo
chart = line_chart + points_all + text_labels

st.altair_chart(chart, use_container_width=True)


st.metric("Predicción próxima hora", f"${next_price:.2f}")

# Mostrar métricas en formato agradable
if metrics:
    # Extraemos valores y manejamos si no están disponibles
    rmse_train = metrics.get('rmse_train', None)
    rmse_val = metrics.get('rmse_val', None)
    r2_train = metrics.get('r2_train', None)
    r2_val = metrics.get('r2_val', None)

    def calc_diff(train, test):
        if train is None or test is None:
            return None
        # Porcentaje de diferencia relativa |train - test| / train * 100
        return abs(train - test) / abs(train) * 100 if train != 0 else None

    data = {
        "Metric": ["RMSE", "R2"],
        "Train": [rmse_train, r2_train],
        "Test": [rmse_val, r2_val],
        "Diff (%)": [calc_diff(rmse_train, rmse_val), calc_diff(r2_train, r2_val)],
    }

    df_metrics = pd.DataFrame(data)
    df_metrics["Train"] = df_metrics["Train"].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    df_metrics["Test"] = df_metrics["Test"].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    df_metrics["Diff (%)"] = df_metrics["Diff (%)"].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")

    st.subheader("Resumen de Métricas del Modelo")
    st.table(df_metrics)
else:
    st.write("No se encontraron métricas del modelo.")
    
# Mostrar información del modelo
st.subheader("Información del Modelo")
st.write("Modelo cargado:", model.__class__.__name__)
st.write("Fecha de predicción:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
st.write("Última actualización de datos:", df["Datetime_"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"))
st.write("Número de puntos históricos:", len(df))
st.write("Número de puntos totales (incluida la predicción):", len(df_all))

# Mostrar parámetros del modelo
if params:
    st.subheader("Parámetros del Modelo")
    for key, value in params.items():
        st.write(f"{key}: {value}")

# Mostrar imágenes de artefactos del modelo
if artifact_images:
    st.subheader("Valores reales contra predecidos")
    for img in artifact_images:
        st.image(img, caption=os.path.basename(img))


