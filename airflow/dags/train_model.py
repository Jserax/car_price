import os

import mlflow
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from airflow import Variable


def train_model():
    params = {}
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
    train_data = Variable.get("train_data")
    train = pd.read_csv(
        filepath_or_buffer=train_data,
        storage_options={
            "key": os.environ["ACCESS_KEY"],
            "secret": os.environ["SECRET_KEY"],
            "client_kwargs": {
                "endpoint_url": f'http://{os.environ["S3_ENDPOINT_URL"]}',
                "use_ssl": False,
            },
        },
    )
    test_data = Variable.get("test_data")
    test = pd.read_csv(
        filepath_or_buffer=test_data,
        storage_options={
            "key": os.environ["ACCESS_KEY"],
            "secret": os.environ["SECRET_KEY"],
            "client_kwargs": {
                "endpoint_url": f'http://{os.environ["S3_ENDPOINT_URL"]}',
                "use_ssl": False,
            },
        },
    )
    train_pool = Pool(
        train.drop(columns=["timestamp", "price"]),
        label=train[["price"]],
        cat_features=cat_cols,
    )
    test_pool = Pool(
        test.drop(columns=["timestamp", "price"]),
        label=test[["price"]],
        cat_features=cat_cols,
    )

    model = CatBoostRegressor(logging_level="Silent", **params)
    model.fit(train_pool)
    y_pred = model.predict(test_pool)
    metrics = [
        r2_score(test["price"], y_pred),
        mean_squared_error(test["price"], y_pred),
        mean_absolute_error(test["price"], y_pred),
    ]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("car_price")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("r2_score", metrics[0])
        mlflow.log_metric("mse", metrics[1])
        mlflow.log_metric("mae", metrics[2])
        input_example = train.iloc[[1]]
        mlflow.catboost.log_model(
            model,
            "car_price",
            registered_model_name="car_price",
            input_example=input_example,
        )
