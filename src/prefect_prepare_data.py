import os
import pandas as pd

from prefect import flow, task, get_run_logger
from utils import get_config, filter_df_by_date



@task
def load_and_merge_data(config_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    cfg = get_config(config_path)
    dataset_dir = os.path.join(cfg['prefect_root_data_dir'], 'dataset')
    date_start = cfg['data_params']['date_start']
    date_end = cfg['data_params']['date_end']

    orders_dataset = pd.read_csv(os.path.join(dataset_dir, 'olist_orders_dataset.csv'))
    orders_dataset['purchase_dt'] = pd.to_datetime(
        orders_dataset['order_purchase_timestamp'].str[:10]
    )

    orders_filtered = filter_df_by_date(
        orders_dataset,
        dt_col='order_purchase_timestamp',
        date_filter={'start_date': date_start, 'end_date': date_end},
    )

    orders_filtered['delivery_time'] = (
        pd.to_datetime(orders_filtered['order_delivered_customer_date']) -
        pd.to_datetime(orders_filtered['order_purchase_timestamp'])
    ).dt.days

    threshold = orders_filtered['delivery_time'].quantile(0.95)
    orders_filtered = orders_filtered[orders_filtered['delivery_time'] <= threshold]

    sellers = pd.read_csv(os.path.join(dataset_dir, 'olist_sellers_dataset.csv'))
    customers = pd.read_csv(os.path.join(dataset_dir, 'olist_customers_dataset.csv'))
    items = pd.read_csv(os.path.join(dataset_dir, 'olist_order_items_dataset.csv'))
    locations = pd.read_csv(os.path.join(dataset_dir, 'olist_geolocation_dataset.csv'))

    locations = (
        locations.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']]
        .mean()
        .reset_index()
    )

    df = (
        orders_filtered
        .merge(items[['order_id', 'price', 'seller_id', 'product_id']], on='order_id')
        .merge(sellers[['seller_id', 'seller_zip_code_prefix']], on='seller_id')
        .merge(customers[['customer_id', 'customer_zip_code_prefix']], on='customer_id')
        .merge(
            locations.rename(columns={
                'geolocation_zip_code_prefix': 'customer_zip_code_prefix',
                'geolocation_lat': 'customer_lat',
                'geolocation_lng': 'customer_lng',
            }),
            on='customer_zip_code_prefix'
        )
    )

    df = df[[
        'seller_zip_code_prefix',
        'customer_lat',
        'customer_lng',
        'delivery_time',
        'purchase_dt'
    ]]

    output_path = os.path.join(cfg['prefect_root_data_dir'], 'prefect', 'merged_dataset.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Merged dataset saved to {output_path}")
    return output_path


@task
def split_train_valid(path: str, config_path: str):
    logger = get_run_logger()
    cfg = get_config(config_path)
    df = pd.read_csv(path)
    dt_col = 'purchase_dt'

    train_start = cfg['data_params']['train_date_start']
    train_end = cfg['data_params']['train_date_end']
    valid_start = cfg['data_params']['valid_date_start']
    valid_end = cfg['data_params']['valid_date_end']

    train_path = os.path.join(cfg['prefect_root_data_dir'], 'prefect', 'train_dataset.csv')
    valid_path = os.path.join(cfg['prefect_root_data_dir'], 'prefect', 'valid_dataset.csv')

    df[(df[dt_col] >= train_start) & (df[dt_col] <= train_end)].to_csv(train_path, index=False)
    df[(df[dt_col] >= valid_start) & (df[dt_col] <= valid_end)].to_csv(valid_path, index=False)

    logger.info(f"Train and valid datasets saved to: {train_path}, {valid_path}")


@flow(name="Prepare Data")
def prefect_prepare_data_flow(config_path: str = "src/config.yml"):
    merged = load_and_merge_data(config_path)
    split_train_valid(merged, config_path)


if __name__ == "__main__":
    prefect_prepare_data_flow()
