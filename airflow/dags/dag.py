from datetime import datetime, timedelta

from process_data import get_raw_data, process_data
from train_model import train_model

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

with DAG(
    dag_id="prepare_data",
    start_date=datetime(2000, 1, 1),
    description="Data preparation for model training",
    default_args={"depends_on_past": False, "retries": 1},
    schedule_interval=timedelta(minutes=30),
    catchup=False,
    tags=["critical", "data"],
):
    start_dag = EmptyOperator(task_id="start_dag")
    end_dag = EmptyOperator(task_id="end_dag")
    get_raw_data_ = PythonOperator(python_callable=get_raw_data, task_id="get raw data")
    process_data_ = PythonOperator(python_callable=process_data, task_id="process data")
    train_model_ = PythonOperator(python_callable=train_model, task_id="train model")
    build_docker_image = BashOperator

    (
        start_dag
        >> get_raw_data_
        >> process_data_
        >> train_model_
        >> build_docker_image
        >> end_dag
    )
