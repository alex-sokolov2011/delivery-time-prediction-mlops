from datetime import datetime
from datetime import timedelta

import pandas as pd

from prepare_data import preprocess_orders


def dt(hour, minute=0, second=0):
    return datetime(2017, 1, 1, hour, minute, second)


def test_preprocess_orders():
    data = [
        (dt(0), dt(0) + timedelta(days=1)),  # delivery_time = 1
        (dt(0), dt(0) + timedelta(days=2)),  # delivery_time = 2
        (dt(0), dt(0) + timedelta(days=3)),  # delivery_time = 3
        (dt(0), dt(0) + timedelta(days=4)),  # delivery_time = 4
    ]

    columns = ['order_purchase_timestamp', 'order_delivered_customer_date']
    df = pd.DataFrame(data, columns=columns)

    df_result = preprocess_orders(df, filter_threshold=2)
    print(df_result)
    resul_rows = df_result.shape[0]
    expected_result = 2
    assert resul_rows == expected_result

    expected_delivery_time_sum = 3
    assert df_result['delivery_time'].sum() == expected_delivery_time_sum
