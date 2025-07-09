import os
import sys

import pandas as pd
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from utils import get_config, train_and_save_model

HPO_EXPERIMENT_NAME = "catboost-params-new"

mlflow.set_tracking_uri("http://mlflow_container_ui:5000")
mlflow.set_experiment(HPO_EXPERIMENT_NAME)


def convert_params(params_dict):
    """
    Convert parameter values from strings to appropriate numeric types.
    """
    converted = {}
    for k, v in params_dict.items():
        try:
            if '.' in v:
                converted[k] = float(v)
            else:
                converted[k] = int(v)
        except ValueError:
            converted[k] = v  # fallback: keep as string
    return converted

def train_best_model(train_data_path, config, model_path):
    top_n=5
    client = MlflowClient()
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(f"Experiment '{HPO_EXPERIMENT_NAME}' not found")
    
    run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )[0]
    run_id = run.info.run_id
    best_params_dict = convert_params(run.data.params)

    print("Best parameters:", best_params_dict)
    print("Best run ID:", run.info.run_id)

    with mlflow.start_run(run_id=run_id):
        train_df = pd.read_csv(train_data_path)
        train_and_save_model(train_df, config, best_params_dict, model_path)
        
        mlflow.log_artifact(model_path, artifact_path="model")
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, name="catboost-best-model")
        
    
    print('model logged and registered')

if __name__ == '__main__':
    config = get_config(sys.argv[1])
    root_data_dir = '/srv/data/'
    model_path = os.path.join('/srv/data', config['model_file_name'])
    train_data_path = os.path.join(root_data_dir, 'train_dataset.csv')

    train_best_model(train_data_path, config, model_path)
