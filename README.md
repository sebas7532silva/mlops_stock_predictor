# Proyecto: Predicción de Precio de AAPL con MLOps

## Descripción General

Este proyecto implementa un sistema completo de predicción del precio de las acciones de Apple (AAPL) utilizando datos históricos descargados de Yahoo Finance. Se emplea un modelo Random Forest para la predicción de precios horarios, acompañado de un flujo de trabajo que sigue buenas prácticas de **MLOps** para automatizar, monitorear y versionar tanto el modelo como los datos.

---

## Objetivos

- Predecir el precio horario de cierre de AAPL.
- Implementar un pipeline reproducible de carga, preprocesamiento, entrenamiento y despliegue del modelo.
- Integrar tracking y versionado del modelo, métricas y artefactos con **MLflow**.
- Crear un dashboard interactivo con **Streamlit** para visualizar predicciones y métricas.
- Automatizar tareas con scripts y manejo de jobs programados (.bat y/o tareas programadas).

---

## Tecnologías y Herramientas

- **Python**: Lenguaje principal para desarrollo.
- **scikit-learn**: Para el modelo Random Forest.
- **pandas**: Procesamiento y manipulación de datos.
- **MLflow**: Registro, seguimiento y versionado de experimentos, modelos y artefactos.
- **Streamlit**: Dashboard web para visualización de predicciones.
- **Joblib**: Serialización del modelo.
- **Tareas Programadas / Scripts .bat**: Automatización de ejecución periódica.
- **Yahoo Finance**: Fuente de datos históricos de AAPL.
- **Telegram API**: Notificaciones de estado (opcional).

---

## Estructura del Proyecto

- **app/data/fetcher.py**: Carga datos desde MongoDB o Yahoo Finance.
- **app/data/preprocessor.py**: Preprocesamiento robusto con extracción de características temporales y lags.
- **app/models/trainer.py**: Entrenamiento del modelo, guardado con Joblib y logueo en MLflow.
- **app/models/predictor.py**: Carga del modelo y predicción del siguiente precio.
- **streamlit_app/dashboard.py**: Interfaz visual que muestra datos históricos, predicción y métricas del modelo.
- **train_and_promote.py**: Script principal que orquesta carga de datos, preprocesamiento, entrenamiento y guardado del modelo.
- **mlruns/**: Directorio donde MLflow almacena experimentos, métricas, modelos y artefactos.

---

## Pipeline y Flujo de Trabajo

1. **Carga de Datos**: Se obtiene información de precios de AAPL con marca de tiempo precisa.
2. **Preprocesamiento**: 
   - Conversión de fechas.
   - Extracción de variables temporales (hora, día de la semana).
   - Creación de variables rezagadas (lags) y medias móviles.
3. **Entrenamiento del Modelo**:
   - Se usa Random Forest con hiperparámetros configurables.
   - Se registran métricas de rendimiento (RMSE, R2) y artefactos (gráficas, archivos CSV).
   - Se guarda el modelo serializado y features usados para la predicción futura.
4. **Tracking con MLflow**:
   - Registro del experimento y ejecución.
   - Logueo de métricas, parámetros, modelo y archivos auxiliares.
5. **Predicción y Visualización**:
   - Carga del último modelo y datos más recientes.
   - Predicción del próximo precio horario.
   - Visualización histórica y predicción en Streamlit.
6. **Automatización**:
   - Scripts .bat o tareas programadas ejecutan el pipeline de entrenamiento cada hora/día.
   - Notificaciones opcionales vía Telegram para seguimiento del proceso.

---

## Buenas Prácticas de MLOps aplicadas

- **Reproducibilidad**: Código modular y versionado con control de versiones.
- **Versionado de modelos y datos**: Uso de MLflow para registrar todas las ejecuciones, parámetros, métricas y artefactos.
- **Monitoreo y alertas**: Integración con Telegram para notificar el éxito o errores del entrenamiento.
- **Despliegue sencillo**: Modelo empaquetado y listo para cargar en producción vía Streamlit.
- **Separación de responsabilidades**: Distinto código para carga, preprocesamiento, entrenamiento, predicción y visualización.
- **Automatización**: Jobs programados para entrenar periódicamente y mantener el modelo actualizado.
- **Escalabilidad**: Diseño para incorporar nuevas fuentes de datos, modelos alternativos o métricas adicionales.

---

## Cómo ejecutar

### Entrenamiento

```bash
python train_and_promote.py
