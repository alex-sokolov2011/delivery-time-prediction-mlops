import os
import sys

from catboost import CatBoostRegressor

from utils import read_data, save_data, get_config, get_features

if __name__ == '__main__':

    config = get_config(sys.argv[1])

    model_file_name = config['model_file_name']
    model_path = os.path.join('/srv/data', model_file_name)

    model = CatBoostRegressor()
    model.load_model(model_path)

    data_path = sys.argv[2]
    output_data_path = sys.argv[3]

    df = read_data(data_path, start_dt=None, end_dt=None)
    X, _ = get_features(df)
    y_pred = model.predict(X)
    X['prediction'] = y_pred
    save_data(X, output_data_path)
