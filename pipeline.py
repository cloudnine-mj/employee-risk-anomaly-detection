import logging
from datetime import datetime, timedelta
import pandas as pd

from config import load_config
from metrics import instrumented
from data.db import DBClient
from features.feature_extractor import FeatureExtractor
from features.sequence_features import SequenceAnomalyDetector
from features.classification_features import ClassificationModel
from detector import RiskAnomalyDetector
from alerting import AlertmanagerClient, filter_alerts, FlappingSuppressor, Deduplicator, MuteList
from governance.governance import ModelRegistry, PerformanceMonitor, AuditLogger

@instrumented
def run_pipeline(cfg: dict) -> pd.DataFrame:
    end = datetime.utcnow()
    lookback = cfg.get('lookback_days', 1)
    start = end - timedelta(days=lookback)

    df_raw = DBClient(cfg['db_uri']).fetch(start, end)
    if df_raw.empty:
        logging.warning("No activity logs for the given window")
        return pd.DataFrame()

    # 기본 피처 추출
    fe = FeatureExtractor(
        resource_weights=cfg.get('resource_weights', {}),
        work_hours=(cfg.get('work_hours_start'), cfg.get('work_hours_end')),
        enable_off_hours_ratio=cfg.get('enable_off_hours_ratio', True)
    )
    X = fe.transform(df_raw)

    # 이상 탐지
    det = RiskAnomalyDetector(
        model_path=cfg['model_path'],
        contamination=cfg['contamination'],
        n_estimators=cfg.get('n_estimators', 100),
        random_state=cfg.get('random_state', 42)
    )
    det.load_or_train(X)
    results = det.detect(X)

    # 시퀀스 이상 탐지
    if cfg.get('enable_sequence_detection', False):
        seq_model = SequenceAnomalyDetector(cfg)
        seq_results = seq_model.run(df_raw)
        seq_results.to_csv(cfg.get('sequence_output_csv', 'sequence_anomalies.csv'))

    # 분류 모델 기반 탐지
    if cfg.get('enable_classification', False):
        clf_model = ClassificationModel(cfg)
        clf_results, shap_df = clf_model.run(df_raw)
        clf_results.to_csv(cfg.get('classifier_output_csv', 'classification_results.csv'))
        shap_df.to_csv(cfg.get('classifier_shap_output_csv', 'classification_shap.csv'))

    # 알림 생성 및 전송
    anomalies = results[results['anomaly'] == 1]
    alerts = [
        {'labels': {'alertname': 'RiskAnomaly', 'user_id': uid},
         'annotations': {'summary': 'User risk anomaly detected'},
         'startsAt': datetime.utcnow().isoformat() + 'Z'}
        for uid in anomalies.index
    ]
    flapper = FlappingSuppressor(
        timedelta(minutes=cfg.get('flap_window_minutes', 5)),
        threshold=cfg.get('flap_threshold', 3)
    )
    deduper = Deduplicator(
        timedelta(minutes=cfg.get('dedup_window_minutes', 10))
    )
    mutes = MuteList(cfg.get('mute_rules', []))
    filtered = filter_alerts(alerts, flapper, deduper, mutes)
    if cfg.get('alertmanager_url'):
        client = AlertmanagerClient(cfg['alertmanager_url'])
        client.send_alerts(filtered)

    # 모델 버전 관리
    if cfg.get('mlflow_tracking_uri'):
        registry = ModelRegistry(cfg['mlflow_tracking_uri'], cfg.get('mlflow_experiment', 'AnomalyDetection'))
        run_id = registry.log_training(
            model=det.model,
            params={'contamination': cfg['contamination']},
            metrics={'num_samples': len(X), 'num_anomalies': int(results['anomaly'].sum())}
        )
        logging.info(f"Logged model run {run_id} to MLflow")

    # 성능 평가
    if cfg.get('performance_thresholds') and 'is_risk' in df_raw.columns:
        y_true = df_raw.groupby('user_id')['is_risk'].max().reindex(X.index).fillna(0).astype(int)
        monitor = PerformanceMonitor(cfg['performance_thresholds'])
        p_metrics, p_alerts = monitor.evaluate(y_true, results['anomaly'])
        for alert in p_alerts:
            logging.warning(f"Performance alert: {alert}")

    # 감사 로그 기록
    if cfg.get('audit_db_uri'):
        auditor = AuditLogger(cfg['audit_db_uri'])
        auditor.log_event('detection', {
            'timestamp': datetime.utcnow().isoformat(),
            'num_anomalies': int(results['anomaly'].sum()),
            'params': {'contamination': cfg['contamination']}
        })

    # 결과 저장
    results.to_csv(cfg.get('output_csv', 'anomaly_results.csv'))
    logging.info("Results saved to output_csv")
    return results

if __name__ == '__main__':
    cfg = load_config('config.yaml')
    run_pipeline(cfg)