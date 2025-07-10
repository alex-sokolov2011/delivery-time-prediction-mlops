import os
import logging

import yaml
import pandas as pd
from dotenv import load_dotenv
from catboost import CatBoostRegressor

load_dotenv()
options = {'client_kwargs': {'endpoint_url': 'http://localstack:4566'}}


def get_model(params, categorical=None):
    if categorical is None:
        categorical = [
            'seller_zip_code_prefix',
            'customer_zip_code_prefix',
        ]
    model = CatBoostRegressor(
        cat_features=categorical,
        verbose=0,
        train_dir="/srv/data/catboost_info",
        **params,
    )

    return model


def get_config(config_path=None):
    if config_path is None:
        if os.getenv("CONFIG_PATH") is None:
            config_path = "config.yml"
        else:
            config_path = os.environ["CONFIG_PATH"]

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_features(df):
    categorical = ['seller_zip_code_prefix', 'customer_zip_code_prefix']
    numerical = [
        'delivery_distance_km',
    ]
    target = 'delivery_time'
    X = df[categorical + numerical]
    if target in df.columns:
        y = df[target]
    else:
        y = None
    return X, y


def filter_df_by_date(df, dt_col, date_filter):
    if date_filter is not None:
        start_date = date_filter['start_date']
        end_date = date_filter['end_date']
        mask = (df[dt_col] < start_date) | (df[dt_col] > end_date)
        df = df.drop(df[mask].index)
    return df


def train_and_save_model(train_df, config, params, model_path):
    if os.path.exists(model_path):
        return

    categorical = config['categorical']
    numerical = config['numerical']
    target = 'delivery_time'
    X_train = train_df[categorical + numerical]
    y_train = train_df[target]

    model = get_model(params, categorical)
    print('Start trainig')
    model.fit(X_train, y_train)
    print('Trainig finished')
    model.save_model(model_path)


def read_data(data_path, start_dt, end_dt):
    """NOTE: S3 for the integration tests"""
    if 's3://' in data_path:
        df = pd.read_csv(data_path, compression=None, storage_options=options)
    else:
        df = pd.read_csv(data_path)
    if start_dt is not None and end_dt is not None:
        df = filter_df_by_date(
            df,
            dt_col='order_purchase_timestamp',
            date_filter={'start_date': start_dt, 'end_date': end_dt},
        )
    return df


def save_data(df, output_file):
    if 's3://' in output_file:
        df.to_csv(output_file, compression=None, index=False, storage_options=options)
    else:
        df.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)
