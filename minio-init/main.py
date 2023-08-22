import os
from datetime import datetime, timedelta

import pandas as pd
from minio import Minio
from numpy import random


def make_buckets():
    client = Minio(
        endpoint=os.environ["S3_ENDPOINT_URL"],
        access_key=os.environ["ACCESS_KEY"],
        secret_key=os.environ["SECRET_KEY"],
    )
    client.make_bucket("mlflow")
    client.make_bucket("datalake")
    client.make_bucket("datasets")


def push_data(path: str):
    data = pd.read_parquet("./data/car_prices.parquet", index_col=0)
    current_date = datetime.now().timestamp()
    max_date = (datetime.now() + timedelta(minutes=45)).timestamp()
    random.uniform(current_date, max_date, size=data.shape[0])
    data["timestamp"] = random.uniform(
        current_date, max_date, size=data.shape[0]
    ).astype("int")
    data.to_parquet(
        "s3://datalake/data.parquet",
        index=False,
        storage_options={
            "key": os.environ["ACCESS_KEY"],
            "secret": os.environ["SECRET_KEY"],
            "client_kwargs": {
                "endpoint_url": f'http://{os.environ["S3_ENDPOINT_URL"]}',
                "use_ssl": False,
            },
        },
    )
