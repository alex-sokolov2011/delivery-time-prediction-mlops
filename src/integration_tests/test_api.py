import httpx

def test_fastapi_api():
    payload = {
        "seller_zip_code_prefix": 9350,
        "customer_lat": -23.57698293467452,
        "customer_lng": -46.58716127427677
    }

    response = httpx.post("http://127.0.0.1:8090/delivery_time", json=payload)

    assert response.status_code == 200
    assert "delivery_time" in response.json()