import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from airflow import Variable


def get_raw_data() -> None:
    """save data for a certain interval from the datalake"""

    df = pd.read_parquet(
        filepath_or_buffer="s3://datalake/data.parquet",
        index_col=0,
        storage_options={
            "key": os.environ["ACCESS_KEY"],
            "secret": os.environ["SECRET_KEY"],
            "client_kwargs": {
                "endpoint_url": f'http://{os.environ["S3_ENDPOINT_URL"]}',
                "use_ssl": False,
            },
        },
    )
    start_date = Variable.get("car_last_date", datetime(year=2000).timestamp())
    end_date = datetime.now().timestamp()
    df = df[(df.timestamp > start_date) & (df.timestamp <= end_date)]
    date = datetime.fromtimestamp(end_date).strftime("%d-%m-%y_%H:%M")
    Variable.set("car_last_date", end_date)
    raw_data = f"s3://datasets/raw_{date}.parquet"
    Variable.set("raw_data", raw_data)
    df.to_parquet(
        raw_data,
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


def process_data() -> None:
    """process raw data, split and save"""

    def cmn(x: pd.Series) -> np.float:
        try:
            return x.mode()[0]
        except:
            return np.nan

    def process_split(
        split: pd.DataFrame, num_cols: list, cat_cols: list
    ) -> pd.DataFrame:
        df = split.drop(columns=["vehicleConfiguration", "link", "parse_date"])
        df.engineDisplacement = (
            df.engineDisplacement.str.split(" ").str[0].astype("float")
        )

        for col in num_cols:
            df[col] = df[col].fillna(
                df.groupby(["brand", "name"])[col]
                .transform("median")
                .fillna(df[col].median())
            )

        for col in cat_cols:
            df[col] = df[col].fillna(
                df.groupby(["brand", "name"])[col]
                .transform(cmn)
                .fillna(df[col].mode()[0])
            )
        return df

    date = Variable.get("car_last_date")
    date = datetime.fromtimestamp(date).strftime("%d-%m-%y_%H:%M")
    raw_data = Variable.get("raw_data")
    df = pd.read_parquet(
        filepath_or_buffer=raw_data,
        storage_options={
            "key": os.environ["ACCESS_KEY"],
            "secret": os.environ["SECRET_KEY"],
            "client_kwargs": {
                "endpoint_url": f'http://{os.environ["S3_ENDPOINT_URL"]}',
                "use_ssl": False,
            },
        },
    )
    num_cols = ["year", "mileage", "power", "engineDisplacement"]
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
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    train = process_split(split=train, num_cols=num_cols, cat_cols=cat_cols)
    test = process_split(split=test, num_cols=num_cols, cat_cols=cat_cols)

    train_data = f"s3://datasets/train_{date}.parquet"
    Variable.set("train_data", train_data)
    df.to_parquet(
        train_data,
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
    test_data = f"s3://datasets/test_{date}.parquet"
    Variable.set("test_data", test_data)
    df.to_parquet(
        test_data,
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
