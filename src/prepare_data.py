import os
import sys

import numpy as np
import pandas as pd

from utils import get_config, filter_df_by_date


def preprocess_orders(df, filter_threshold=None):
    df['delivery_time'] = (
        pd.to_datetime(df['order_delivered_customer_date']) -
        pd.to_datetime(df['order_purchase_timestamp'])
    ).dt.days
    if filter_threshold is None:
        filter_threshold = df['delivery_time'].quantile(0.95)
    df = df[df['delivery_time'] <= filter_threshold]
    return df


def prepare_data(root_dir, start_date, end_date, dataset_dir=None):
    result_csv_path = os.path.join(root_dir, 'merged_dataset.csv')

    if os.path.exists(result_csv_path):
        return result_csv_path

    orders_dataset = pd.read_csv(os.path.join(dataset_dir, 'olist_orders_dataset.csv'))
    orders_dataset['purchase_dt'] = pd.to_datetime(
        orders_dataset['order_purchase_timestamp'].apply(lambda x: x[:10])
    )

    orders_filtered_df = filter_df_by_date(
        orders_dataset,
        dt_col='order_purchase_timestamp',
        date_filter={'start_date': start_date, 'end_date': end_date},
    )
    orders_filtered_df = preprocess_orders(orders_filtered_df)
    #
    sellers_df = pd.read_csv(os.path.join(dataset_dir, 'olist_sellers_dataset.csv'))
    customers_df = pd.read_csv(os.path.join(dataset_dir, 'olist_customers_dataset.csv'))
    locations_df = pd.read_csv(
        os.path.join(dataset_dir, 'olist_geolocation_dataset.csv')
    )
    orders_items_dataset = pd.read_csv(
        os.path.join(dataset_dir, 'olist_order_items_dataset.csv')
    )

    locations_df = (
        locations_df.groupby('geolocation_zip_code_prefix')[
            ['geolocation_lat', 'geolocation_lng']
        ]
        .mean()
        .reset_index()
    )

    delivery_df = (
        orders_filtered_df[
            [
                'order_id',
                'purchase_dt',
                'customer_id',
                'order_delivered_customer_date',
                'order_purchase_timestamp',
                'delivery_time',
            ]
        ]
        .merge(
            orders_items_dataset[['order_id', 'price', 'seller_id', 'product_id']],
            on='order_id',
        )
        .merge(sellers_df[['seller_id', 'seller_zip_code_prefix']], on='seller_id')
        .merge(
            customers_df[['customer_id', 'customer_zip_code_prefix']], on='customer_id'
        )
        .merge(
            locations_df[
                ['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']
            ].rename(
                columns={
                    'geolocation_zip_code_prefix': 'customer_zip_code_prefix',
                    'geolocation_lat': 'customer_lat',
                    'geolocation_lng': 'customer_lng'
                }
            ),
            on='customer_zip_code_prefix',
        )

    )
    delivery_df = delivery_df[[
        'seller_zip_code_prefix',
        'customer_lat',
        'customer_lng',
        'delivery_time',
        'purchase_dt'
    ]]
    
    delivery_df.to_csv(result_csv_path, index=False)
    return result_csv_path


def prepare_train_test(input_csv_path, config):
    train_csv_path = os.path.join(config['root_data_dir'], 'train_dataset.csv')
    valid_csv_path = os.path.join(config['root_data_dir'], 'valid_dataset.csv')
    if os.path.exists(train_csv_path) and os.path.exists(valid_csv_path):
        print("Train and validation datasets already exist. Skipping split.")
        return
    else:
        print("Generating new train/valid datasets...")
    df = pd.read_csv(input_csv_path)
    dt_col = 'purchase_dt'

    dt_start = config['data_params']['train_date_start']
    dt_end = config['data_params']['train_date_end']

    mask = (df[dt_col] >= dt_start) | (df[dt_col] <= dt_end)
    df[mask].to_csv(train_csv_path, index=False)

    dt_start = config['data_params']['valid_date_start']
    dt_end = config['data_params']['valid_date_end']
    mask = (df[dt_col] >= dt_start) | (df[dt_col] >= dt_end)
    df[mask].to_csv(valid_csv_path, index=False)
    print('Train test split complited')


if __name__ == '__main__':
    cfg = get_config(sys.argv[1])
    dataset_directory = os.path.join(cfg['root_data_dir'], 'dataset')

    date_start = cfg['data_params']['date_start']
    date_end = cfg['data_params']['date_end']
    result_path = prepare_data(
        cfg['root_data_dir'], date_start, date_end, dataset_dir=dataset_directory
    )

    prepare_train_test(result_path, config=cfg)
