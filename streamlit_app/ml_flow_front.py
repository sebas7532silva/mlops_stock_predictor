import mlflow
import os

def get_last_metrics(experiment_name="AAPL_Prediction"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return {}

    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return {}

    last_run = runs[0]
    metrics = last_run.data.metrics
    return metrics

def get_model_parameters(experiment_name="AAPL_Prediction"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return {}

    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return {}

    last_run = runs[0]
    params = last_run.data.params
    return params

def get_model_artifacts(experiment_name="AAPL_Prediction"):
    import mlflow

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return []

    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return []

    run_id = runs[0].info.run_id
    artifacts = client.list_artifacts(run_id)

    image_files = [f.path for f in artifacts if f.path.endswith(".png")]

    # Carpeta destino donde guardar las imágenes
    images_dir = os.path.join("streamlit_app", "imagenes")
    os.makedirs(images_dir, exist_ok=True)

    downloaded_paths = []
    for path in image_files:
        # Descargar cada artifact en la carpeta images_dir
        local_path = client.download_artifacts(run_id, path, dst_path=images_dir)
        downloaded_paths.append(local_path)

    return downloaded_paths

