import mlflow
import logging

class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_model(self, model, model_path: str, params: dict, metrics: dict, tags: dict = None):
        with mlflow.start_run() as run:
            logging.info(f"Logging model to MLflow: {model_path}")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(model_path, artifact_path="model_artifacts")
            logging.info(f"MLflow Run ID: {run.info.run_id}")
