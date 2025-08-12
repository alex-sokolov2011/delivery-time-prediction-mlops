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
    DatasetMissingValuesMetric,
    ColumnValueRangeMetric,
    ColumnCorrelationsMetric,
)

from utils import read_data, get_config, get_features

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists model_metrics;
drop table if exists public.model_metrics;
create table public.model_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	value_range_share_in_range float,
	prediction_corr_with_features float
)
"""

def get_features(df, config):
    categorical = config['categorical']
    numerical = config['numerical']
    target = 'delivery_time'
    X = df[categorical + numerical]
    y = df[target] if target in df.columns else None
    return X, y

def generate_date_ranges(start_date, end_date):
    # Create a date range for each Monday start within the specified range
    week_starts = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    week_ends = week_starts + pd.Timedelta(days=6)
    return list(zip(week_starts, week_ends))


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
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    value_range_result = result['metrics'][3]['result']['reference']
    value_range_share_in_range = value_range_result.get('share_in_range', 0.0)

    correlations_result = result['metrics'][4]['result']
    pearson_y = correlations_result.get('current', {}).get('pearson', {}).get('values', {}).get('y', [])

    if pearson_y:
        prediction_corr_with_features = sum(abs(val) for val in pearson_y) / len(pearson_y)
    else:
        prediction_corr_with_features = 0.0

    curr.execute(
        "insert into public.model_metrics("
        "timestamp, prediction_drift, num_drifted_columns, share_missing_values, value_range_share_in_range, prediction_corr_with_features"
        ") values (%s, %s, %s, %s, %s, %s)",
        (
            end_date,
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            value_range_share_in_range,
            prediction_corr_with_features,
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
            ColumnValueRangeMetric(column_name='prediction', left=0, right=28),
            ColumnCorrelationsMetric(column_name='prediction'),
        ]
    )


    reference_data_path = os.path.join('/srv/data', 'valid_dataset.csv')
    reference_data_df = pd.read_csv(reference_data_path)
    X, _ = get_features(reference_data_df, config)
    reference_data_df['prediction'] = model.predict(X)

    prep_db()

    for start, end in pairs:
        df = read_data(data_path, f'{start.date()}', f'{end.date()}')
        X, _ = get_features(df, config)
        y_pred = model.predict(X)  
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
