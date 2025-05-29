from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import os
import sys

# 현재 디렉토리에 anomaly_detection 코드 경로 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

from pipeline import run_pipeline
from config import load_config

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['alert@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='combined_anomaly_classification_sequence_pipeline',
    default_args=default_args,
    description='통합 이상 탐지 + 분류 + 시퀀스 기반 리스크 탐지 파이프라인',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['risk', 'anomaly', 'classification', 'sequence'],
) as dag:

    def run_full_pipeline():
        try:
            cfg = load_config('/opt/airflow/config/config.yaml')  # 경로는 Airflow 컨테이너 기준
            result_df = run_pipeline(cfg)
            logging.info(f"Pipeline completed. Result shape: {result_df.shape}")
        except Exception as e:
            logging.exception("Pipeline execution failed")
            raise e

    run_pipeline_task = PythonOperator(
        task_id='run_risk_detection_pipeline',
        python_callable=run_full_pipeline,
    )

    run_pipeline_task
