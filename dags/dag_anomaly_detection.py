from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

from config import load_config
from pipeline import run_pipeline

# 기본 설정
DEFAULT_ARGS = {
    'owner': 'compliance-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# DAG 정의
with DAG(
    dag_id='anomaly_full_pipeline',
    default_args=DEFAULT_ARGS,
    description='정기 리스크 및 시퀀스 이상 탐지 파이프라인',
    schedule_interval='@daily',        # 매일 한 번 실행
    catchup=False,
    max_active_runs=1,
    tags=['anomaly', 'risk', 'sequence']
) as dag:

    def run_full_pipeline(**context):
        logging.info("Loading configuration and running pipeline")
        cfg = load_config('/path/to/config.yaml')
        # run_pipeline 내에 리스크 + 시퀀스 탐지 로직이 모두 포함되어 있습니다
        run_pipeline(cfg)

    run_detection = PythonOperator(
        task_id='run_anomaly_full_pipeline',
        python_callable=run_full_pipeline,
        provide_context=True,
    )

    run_detection