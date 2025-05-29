from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

from config import load_config
from pipeline import run_pipeline
from classification import ClassificationModel
from reporting.report_automation import ReportGenerator, run_dashboard
from db import DBClient
from features import FeatureExtractor
import pandas as pd

# 기본 DAG 설정 
default_args = {
    'owner': 'compliance-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# 태스크 정의 
def task_anomaly(**kwargs):
    cfg = load_config('/path/to/config.yaml')
    logging.info('Running full anomaly detection pipeline')
    run_pipeline(cfg)


def task_classification(**kwargs):
    cfg = load_config('/path/to/config.yaml')
    logging.info('Running classification model pipeline')
    # 데이터 로드 및 피처
    end = datetime.utcnow()
    start = end - timedelta(days=cfg.get('lookback_days', 1))
    df = DBClient(cfg['db_uri']).fetch(start, end)
    fe = FeatureExtractor(resource_weights=cfg.get('resource_weights', {}))
    X = fe.transform(df)
    # 라벨(감사결과) 로드 (is_risk 컬럼 필요)
    y = df.groupby('user_id')['is_risk'].max().reindex(X.index).fillna(0).astype(int)
    # 분류 모델 학습 및 예측
    clf = ClassificationModel(
        model_path=cfg.get('classifier_model_path','classifier_model.joblib'),
        params=cfg.get('classifier_params', {})
    )
    clf.train(X, y)
    preds = clf.predict(X)
    preds.to_csv(cfg.get('classification_output_csv','classification_results.csv'))


def task_reporting(**kwargs):
    cfg = load_config('/path/to/config.yaml')
    logging.info('Generating monthly reports')
    # 이상 탐지 요약
    df_anom = pd.read_csv(cfg['output_csv'], index_col=0)
    df_seq  = pd.read_csv(cfg['sequence_output_csv'], index_col=0) if cfg.get('enable_sequence_detection') else pd.DataFrame()
    df_cls  = pd.read_csv(cfg['classification_output_csv'], index_col=0) if cfg.get('enable_classification') else pd.DataFrame()
    summary = pd.DataFrame({
        'total_anomalies': [len(df_anom)],
        'sequence_anomalies': [len(df_seq)],
        'classified_risks': [int(df_cls['prediction'].sum()) if not df_cls.empty else 0]
    })
    gen = ReportGenerator(template_pptx=cfg.get('pptx_template'))
    gen.generate_pptx(summary, cfg.get('monthly_pptx','monthly_report.pptx'))
    gen.generate_excel(summary, cfg.get('monthly_excel','monthly_report.xlsx'))


def task_dashboard(**kwargs):
    cfg = load_config('/path/to/config.yaml')
    logging.info('Launching Streamlit dashboard')
    run_dashboard(cfg.get('sequence_output_csv','sequence_anomalies.csv'))

# DAG 정의
with DAG(
    dag_id='full_anomaly_pipeline',
    default_args=default_args,
    description='Risk detection → Classification → Reporting → Dashboard',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['anomaly','classification','reporting','dashboard']
) as dag:
    t1 = PythonOperator(
        task_id='anomaly_detection',
        python_callable=task_anomaly,
        provide_context=True
    )
    t2 = PythonOperator(
        task_id='classification_model',
        python_callable=task_classification,
        provide_context=True
    )
    t3 = PythonOperator(
        task_id='generate_reports',
        python_callable=task_reporting,
        provide_context=True
    )
    t4 = PythonOperator(
        task_id='launch_dashboard',
        python_callable=task_dashboard,
        provide_context=True
    )

    # 순차 의존성
    t1 >> t2 >> t3 >> t4
