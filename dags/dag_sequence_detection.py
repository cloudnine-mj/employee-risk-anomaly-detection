from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from config import load_config
from sequence_detector import SequenceAnomalyDetector
from db import DBClient
import pandas as pd
import logging

DEFAULT_ARGS = {
    'owner': 'compliance-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

def run_sequence_detection(**ctx):
    cfg = load_config('/path/to/config.yaml')
    # DB 로드
    end = datetime.utcnow()
    start = end - timedelta(days=cfg.get('lookback_days', 1))
    df = DBClient(cfg['db_uri']).fetch(start, end)
    if df.empty:
        logging.info("No data for sequence detection")
        return

    seq_detector = SequenceAnomalyDetector(
        embed_size=cfg.get('embed_size', 50),
        lstm_units=cfg.get('lstm_units', 32),
        max_seq_len=cfg.get('max_seq_len', 100),
        epochs=cfg.get('seq_epochs', 10),
        batch_size=cfg.get('seq_batch_size', 32),
        contamination=cfg.get('contamination', 0.01)
    )
    seq_detector.train(df)
    results = seq_detector.detect(df)
    out_file = cfg.get('sequence_output_csv', 'sequence_anomalies.csv')
    results.to_csv(out_file)
    logging.info(f"Sequence anomalies written to {out_file}")

with DAG(
    'sequence_anomaly_detection',
    default_args=DEFAULT_ARGS,
    schedule_interval='@daily',
    catchup=False,
    tags=['anomaly','sequence'],
) as dag:

    detect_task = PythonOperator(
        task_id='run_sequence_detection',
        python_callable=run_sequence_detection,
        provide_context=True,
    )

    detect_task
