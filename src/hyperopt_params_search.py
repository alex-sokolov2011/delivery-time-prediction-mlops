import os
import sys

import pandas as pd
import numpy as np

import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import root_mean_squared_error

from utils import get_config, get_model, get_features

mlflow.set_tracking_uri("http://mlflow_container_ui:5000")
experiment_name = "catboost-params-new"

try:
    mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")
except Exception:
    pass  # Experiment already exists

mlflow.set_experiment(experiment_name)


def run_optimization(root_data_dir: str, num_trials: int):
    train_data_path = os.path.join(root_data_dir, 'train_dataset.csv')
    valid_data_path = os.path.join(root_data_dir, 'valid_dataset.csv')
    artefact_model_path = os.path.join(root_data_dir,  'artefact_model', 'catboost_model.cbm')
    os.makedirs(os.path.dirname(artefact_model_path), exist_ok=True)
    
    # Prepare the training data
    train_df = pd.read_csv(train_data_path)
    X_train, y_train = get_features(train_df)

    valid_df = pd.read_csv(valid_data_path)
    X_valid, y_valid = get_features(valid_df)

    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            model = get_model(params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            rmse = root_mean_squared_error(y_valid, y_pred)
            mlflow.log_metric("rmse", rmse)

            model.save_model(artefact_model_path)
            mlflow.log_artifact(artefact_model_path, artifact_path="model")
        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'iterations': hp.quniform('iterations', 100, 500, 50),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'depth': hp.quniform('depth', 4, 10, 1),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10)
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    root_data_dir = '/srv/data/'
    config = get_config(sys.argv[1])

    run_optimization(root_data_dir, num_trials=15)
