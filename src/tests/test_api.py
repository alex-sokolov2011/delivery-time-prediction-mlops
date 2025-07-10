import httpx

def test_fastapi_api():
    payload = {
        "seller_zip_code_prefix": 9350,
        "customer_zip_code_prefix": 3149,
        "delivery_distance_km": 18
    }

    response = httpx.post("http://127.0.0.1:8090/delivery_time", json=payload)

    assert response.status_code == 200
    assert "delivery_time" in response.json()