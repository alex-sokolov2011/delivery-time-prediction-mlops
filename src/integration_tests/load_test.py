import os
import sys
import time
import httpx
import random
import requests
import pandas as pd



if __name__ == '__main__':
    # Load config

    # Path to validation dataset
    data_path = os.path.join('/srv/data', 'merged_dataset.csv')

    # Path to save results
    output_path =os.path.join('/srv/data', 'load_test_results.csv')

    # Load validation data and take 1000 random rows
    df = pd.read_csv(data_path)
    features = df.dropna().sample(1000, random_state=42)

    url = "http://172.17.0.1:8090/delivery_time"
    end_time = time.time() + 600  # 10 minutes
    results = []

    print("Starting load test for 10 minutes...")

    while time.time() < end_time:
        row = features.sample(1).iloc[0]

        payload = {
            "seller_zip_code_prefix": int(row.seller_zip_code_prefix),
            "customer_lat": float(row.customer_lat),
            "customer_lng": float(row.customer_lng),
        }

        try:
            r = httpx.post(url, json=payload)
            pred = r.json().get("delivery_time", None)
            print(f"Payload: {payload}, Prediction: {pred}")
            results.append(
                {
                    "seller_zip_code_prefix": payload["seller_zip_code_prefix"],
                    "customer_lat": payload["customer_lat"],
                    "customer_lng": payload["customer_lng"],
                    "prediction": pred,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            results.append(
                {
                    "seller_zip_code_prefix": payload["seller_zip_code_prefix"],
                    "customer_lat": payload["customer_lat"],
                    "customer_lng": payload["customer_lng"],
                    "prediction": None,
                    "timestamp": time.time(),
                    "error": str(e),
                }
            
            )
            print(f"Error for payload {payload}: {e}")

        # Random delay between 0.1–1.0 sec
        time.sleep(random.uniform(5, 15))

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Load test finished. Results saved to {output_path}")
