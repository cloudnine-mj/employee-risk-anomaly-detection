from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

from config import load_config
from data.db import DBClient
from features.features import FeatureExtractor
from detector.classification import ClassificationModel
import pandas as pd

# 기본 DAG 인수 설정
DEFAULT_ARGS = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def run_classification_pipeline(**context):
    # 설정 로드
    cfg = load_config('/path/to/config.yaml')
    logging.info(f"Loaded config for classification: {cfg}")

    # 데이터 로드
    end = datetime.utcnow()
    start = end - timedelta(days=cfg.get('lookback_days', 1))
    df = DBClient(cfg['db_uri']).fetch(start, end)
    if df.empty:
        logging.warning("No data to train classification model")
        return

    # 피처 추출
    fe = FeatureExtractor(resource_weights=cfg.get('resource_weights', {}))
    X = fe.transform(df)

    # 라벨 로드 (예: audit 결과 컬럼)
    if 'is_risk' not in df.columns:
        raise ValueError("Raw data must include 'is_risk' column for supervised training")
    y = df.groupby('user_id')['is_risk'].max().reindex(X.index).fillna(0).astype(int)

    # 분류 모델 학습 및 평가
    clf = ClassificationModel(
        model_path=cfg.get('classifier_model_path', 'classifier_model.joblib'),
        params=cfg.get('classifier_params', {})
    )
    clf.train(X, y)

    # 예측 및 결과 저장
    preds = clf.predict(X)
    out_path = cfg.get('classification_output_csv', 'classification_results.csv')
    preds.to_csv(out_path)
    logging.info(f"Classification results saved to {out_path}")

    # SHAP 설명 저장
    if clf.explainer:
        shap_df = clf.explain(X, max_display=cfg.get('classifier_shap_max_display', 10))
        shap_out = cfg.get('classification_shap_output_csv', 'classification_shap.csv')
        shap_df.to_csv(shap_out)
        logging.info(f"SHAP values saved to {shap_out}")


default_args = DEFAULT_ARGS
with DAG(
    dag_id='classification_model_pipeline',
    default_args=default_args,
    description='Supervised classification model training and prediction',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['classification', 'risk']
) as dag:

    classify = PythonOperator(
        task_id='run_classification_pipeline',
        python_callable=run_classification_pipeline,
        provide_context=True
    )

    classify
