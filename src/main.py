from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import pandas as pd

app = FastAPI()

model = CatBoostRegressor()
model.load_model("/srv/data/prod_model.cbm")

class DeliveryTimeRequest(BaseModel):
    seller_zip_code_prefix: int
    customer_zip_code_prefix: int
    delivery_distance_km: int

class DeliveryTimeResponse(BaseModel):
    seller_zip_code_prefix: int
    customer_zip_code_prefix: int
    delivery_distance_km: int
    delivery_time: int

@app.post("/delivery_time", response_model=DeliveryTimeResponse)
async def delivery_time(request: DeliveryTimeRequest):
    try:
        seller_zip_code_prefix = request.seller_zip_code_prefix
        customer_zip_code_prefix = request.customer_zip_code_prefix
        delivery_distance_km = request.delivery_distance_km

        X = pd.DataFrame([{
            "seller_zip_code_prefix": seller_zip_code_prefix,
            "customer_zip_code_prefix": customer_zip_code_prefix,
            "delivery_distance_km": delivery_distance_km,
        }])
        delivery_time = int(round(model.predict(X)[0]))

        # Return the result as JSON
        return DeliveryTimeResponse(
            seller_zip_code_prefix=seller_zip_code_prefix,
            customer_zip_code_prefix=customer_zip_code_prefix,
            delivery_distance_km=delivery_distance_km,
            delivery_time=delivery_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
