import os
import sys
import random

import pandas as pd
import psycopg
from catboost import CatBoostRegressor
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    ColumnQuantileMetric,
    ColumnValueListMetric,
    DatasetMissingValuesMetric,
)

from utils import read_data, get_config, get_features

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists model_metrics;
create table model_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	delivery_distance_km_q95 float,
	most_frequent_seller_value_share float
)
"""


def generate_date_ranges(start_date, end_date):
    # Create a date range for each month start within the specified range
    month_starts = pd.date_range(start=start_date, end=end_date, freq='MS')
    month_ends = pd.date_range(start=start_date, end=end_date, freq='M')

    # Return pairs of start and end dates
    return list(zip(month_starts, month_ends))


def prep_db():
    with psycopg.connect(
        "host=db port=5432 user=db_user password=db_password", autocommit=True
    ) as pg_conn:
        res = pg_conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            pg_conn.execute("create database test;")
        with psycopg.connect(
            "host=db port=5432 dbname=test user=db_user password=db_password"
        ) as conn:
            conn.execute(create_table_statement)


def calculate_metrics_postgresql(
    curr, current_data, end_date, reference_data, report, col_mapping
):
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=col_mapping,
    )

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current'][
        'share_of_missing_values'
    ]
    delivery_distance_km_q95 = result['metrics'][3]['result']['current']['value']
    most_frequent_seller_value_share = result['metrics'][4]['result']['current'][
        'values_in_list_dist'
    ]['y'][0] / sum(
        i for i in result['metrics'][4]['result']['current']['values_in_list_dist']['y']
    )

    curr.execute(
        "insert into model_metrics("
        "timestamp, prediction_drift, num_drifted_columns, share_missing_values, "
        "delivery_distance_km_q95, most_frequent_seller_value_share"
        ") values (%s, %s, %s, %s, %s, %s)",
        (
            end_date,
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            delivery_distance_km_q95,
            most_frequent_seller_value_share,
        ),
    )


if __name__ == '__main__':
    config = get_config(sys.argv[1])

    start_dt = config['data_params']['backfill_date_start']
    end_dt = config['data_params']['backfill_date_end']
    pairs = generate_date_ranges(start_dt, end_dt)
    model_file_name = config['model_file_name']
    model_path = os.path.join('/srv/data', model_file_name)
    data_path = os.path.join('/srv/data', 'merged_dataset.csv')

    # params = config['catboost_params']
    # model = get_model(params)

    model = CatBoostRegressor()
    model.load_model(model_path)

    ev_column_mapping = ColumnMapping(
        prediction='prediction',
        numerical_features=config['numerical'],
        categorical_features=config['categorical'],
        target=None,
    )

    ev_report = Report(
        metrics=[
            ColumnDriftMetric(column_name='prediction'),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnQuantileMetric(column_name='delivery_distance_km', quantile=0.95),
            ColumnValueListMetric(column_name='seller_zip_code_prefix'),
        ]
    )

    reference_data_path = os.path.join('/srv/data', 'valid_dataset.csv')
    reference_data_df = pd.read_csv(reference_data_path)
    X, _ = get_features(reference_data_df)
    reference_data_df['prediction'] = model.predict(X)

    prep_db()

    for start, end in pairs:
        df = read_data(data_path, f'{start.date()}', f'{end.date()}')
        X, _ = get_features(df)
        y_pred = model.predict(X)  # batch_prediction_backfill(start_date, end_date)
        df['prediction'] = y_pred
        with psycopg.connect(
            "host=db port=5432 dbname=test user=db_user password=db_password",
            autocommit=True,
        ) as conn:
            with conn.cursor() as cursor:
                calculate_metrics_postgresql(
                    cursor, df, end, reference_data_df, ev_report, ev_column_mapping
                )
        num_rows = df.shape[0]
        print(
            f"Start: {start.date()}, End: {end.date()}, num_rows {num_rows}, "
            f"prediction mean: {y_pred.mean():.4f}"
        )
