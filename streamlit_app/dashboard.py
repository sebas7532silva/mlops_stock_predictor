import streamlit as st
import pandas as pd
import altair as alt
import sys
import os

# Importar funciones y modelos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.models.predictor import load_latest_model, predict_next_price, load_latest_data
from ml_flow_front import get_last_metrics, get_model_parameters, get_model_artifacts

st.set_page_config(page_title="Dashboard AAPL", layout="wide")
st.title("📊 Dashboard de Predicción: AAPL Stock Price")
st.caption("Análisis de series temporales y predicción de precios horarios para Apple Inc.")

# ======================= Cargar modelo, datos y métricas =======================
model = load_latest_model()
metrics = get_last_metrics("AAPL_Prediction")
params = get_model_parameters("AAPL_Prediction")
artifact_images = get_model_artifacts("AAPL_Prediction")

df = load_latest_data()
df = df.sort_values("Datetime_").tail(5).copy()

# ======================= Predicción del siguiente punto =======================
next_price = predict_next_price(model)
last_datetime = df["Datetime_"].iloc[-1]
next_datetime = last_datetime + pd.Timedelta(hours=1)

pred_row = pd.DataFrame({
    "Datetime_": [next_datetime],
    "Close_AAPL": [next_price]
})
df_all = pd.concat([df, pred_row], ignore_index=True)
df_all["segment"] = "histórico"
df_all.loc[df_all.index[-2:], "segment"] = ["histórico", "predicción"]

# ======================= Tabs =======================
tab1, tab2, tab3 = st.tabs(["📈 Predicción", "📏 Métricas del Modelo", "🖼️ Artefactos"])

# ======================= TAB 1: Predicción =======================
with tab1:
    st.subheader("🧮 Evolución reciente y próxima predicción")

    y_min = df_all["Close_AAPL"].min() * 0.98
    y_max = df_all["Close_AAPL"].max() * 1.02

    line_chart = alt.Chart(df_all).mark_line().encode(
        x=alt.X('Datetime_:T', title='Fecha y Hora'),
        y=alt.Y('Close_AAPL:Q', title='Precio AAPL', scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color('segment:N',
                        scale=alt.Scale(domain=["histórico", "predicción"],
                                        range=["#1f77b4", "#d62728"]),
                        legend=None)
    ).properties(width=800, height=400)

    points_all = alt.Chart(df_all).mark_point(filled=True, size=90).encode(
        x='Datetime_:T',
        y='Close_AAPL:Q',
        color=alt.Color('segment:N',
                        scale=alt.Scale(domain=["histórico", "predicción"],
                                        range=["#1f77b4", "#d62728"]),
                        legend=None)
    )

    text_labels = alt.Chart(df_all).mark_text(
        align='center',
        baseline='bottom',
        dy=-10,
        fontSize=12,
        fontWeight='bold'
    ).encode(
        x='Datetime_:T',
        y='Close_AAPL:Q',
        text=alt.Text('Close_AAPL:Q', format=".2f"),
        color=alt.Color('segment:N',
                        scale=alt.Scale(domain=["histórico", "predicción"],
                                        range=["#1f77b4", "#d62728"]),
                        legend=None)
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.altair_chart(line_chart + points_all + text_labels, use_container_width=True)

    with col2:
        st.metric(label="🔮 Siguiente Precio Predicho", value=f"${next_price:.2f}")
        st.write(f"**Último dato:** {last_datetime.strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**Nuevo dato predicho:** {next_datetime.strftime('%Y-%m-%d %H:%M')}")

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.write("**Tipo de modelo:**", model.__class__.__name__)
        st.write("**Fecha de ejecución:**", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        st.write("**Última actualización de datos:**", df["Datetime_"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"))

    with col4:
        st.write("**Datos históricos:**", len(df))
        st.write("**Datos totales (con predicción):**", len(df_all))

# ======================= TAB 2: Métricas del Modelo =======================
with tab2:
    st.subheader("📏 Evaluación del Modelo")

    if metrics:
        rmse_train = metrics.get('rmse_train')
        rmse_val = metrics.get('rmse_val')
        r2_train = metrics.get('r2_train')
        r2_val = metrics.get('r2_val')

        def calc_diff(train, test):
            if train is None or test is None:
                return None
            return abs(train - test) / abs(train) * 100 if train != 0 else None

        data = {
            "Métrica": ["RMSE", "R²"],
            "Entrenamiento": [rmse_train, r2_train],
            "Validación": [rmse_val, r2_val],
            "Diferencia (%)": [calc_diff(rmse_train, rmse_val), calc_diff(r2_train, r2_val)],
        }

        df_metrics = pd.DataFrame(data)
        df_metrics["Entrenamiento"] = df_metrics["Entrenamiento"].apply(lambda x: f"{x:.4f}" if x else "N/A")
        df_metrics["Validación"] = df_metrics["Validación"].apply(lambda x: f"{x:.4f}" if x else "N/A")
        df_metrics["Diferencia (%)"] = df_metrics["Diferencia (%)"].apply(lambda x: f"{x:.2f}%" if x else "N/A")

        st.dataframe(df_metrics, use_container_width=True)
    else:
        st.warning("No se encontraron métricas para este modelo.")

    st.divider()

    if params:
        st.subheader("⚙️ Parámetros del Modelo")
        for k, v in params.items():
            st.markdown(f"- **{k}**: `{v}`")

# ======================= TAB 3: Artefactos =======================
with tab3:
    st.subheader("📸 Comparación: Valores Reales vs. Predichos")

    if artifact_images:
        for img in artifact_images:
            col = st.columns(3)[1]  # centrar en la columna del medio
            with col:
                st.image(img, caption=os.path.basename(img))
    else:
        st.info("No se encontraron imágenes de artefactos para este modelo.")



