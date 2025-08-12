import pandas as pd
from fastapi import FastAPI, HTTPException
from catboost import CatBoostRegressor
from pydantic import BaseModel

app = FastAPI()

model = CatBoostRegressor()
model.load_model("/srv/data/prod_model.cbm")


class DeliveryTimeRequest(BaseModel):
    seller_zip_code_prefix: int
    customer_lat: float
    customer_lng: float


class DeliveryTimeResponse(BaseModel):
    seller_zip_code_prefix: int
    customer_lat: float
    customer_lng: float
    delivery_time: int


@app.post("/delivery_time", response_model=DeliveryTimeResponse)
async def delivery_time(request: DeliveryTimeRequest):
    try:
        X = pd.DataFrame(
            [
                {
                    "seller_zip_code_prefix": request.seller_zip_code_prefix,
                    "customer_lat": request.customer_lat,
                    "customer_lng": request.customer_lng,
                }
            ]
        )
        predicted_delivery_time = int(round(model.predict(X)[0]))

        return DeliveryTimeResponse(
            seller_zip_code_prefix=request.seller_zip_code_prefix,
            customer_lat=request.customer_lat,
            customer_lng=request.customer_lng,
            delivery_time=predicted_delivery_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
