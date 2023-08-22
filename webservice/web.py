from catboost import Pool
from fastapi import FastAPI, status
from pandas import DataFrame
from pydantic import BaseModel

app = FastAPI()


class HealthCheck(BaseModel):
    status: str = "OK"


class CarFeatures(BaseModel):
    year: float
    mileage: float
    power: float
    engineDisplacement: str
    brand: str
    name: str
    bodyType: str
    color: str
    fuelType: str
    transmission: str
    location: str
    engineName: str


class Response(BaseModel):
    price: float


@app.on_event("startup")
def startup_event():
    from mlflow.catboost import load_model

    global model
    model = load_model("runs:/")


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")


@app.get(
    "/predict",
    tags=["predict"],
    summary="Predict the price of a car",
    response_description="Return the price of a car (float value)",
    status_code=status.HTTP_200_OK,
    response_model=Response,
)
def predict(input_data: CarFeatures):
    cat_cols = [
        "brand",
        "name",
        "bodyType",
        "color",
        "fuelType",
        "transmission",
        "location",
        "engineName",
    ]
    data = DataFrame(input_data.dict())
    data["engineDisplacement"] = (
        data["engineDisplacement"].str.split(" ").str[0].astype("float")
    )
    data = Pool(data=data, cat_features=cat_cols)
    model.predict(data)
    return {"text": "Sentiment Analysis"}
