import logging
from datetime import datetime, timedelta
import pandas as pd

from config import load_config
from metrics import instrumented
from db import DBClient
from features import FeatureExtractor
from detector import RiskAnomalyDetector
from alerting import AlertmanagerClient, filter_alerts, FlappingSuppressor, Deduplicator, MuteList
from governance import ModelRegistry, PerformanceMonitor, AuditLogger

@instrumented
def run_pipeline(cfg: dict) -> pd.DataFrame:
    """
    전체 리스크 탐지 파이프라인

    1) 데이터 로드
    2) 피처 추출
    3) 모델 학습/로딩 및 이상 탐지
    4) 알림 억제 로직 적용 후 Alertmanager로 전송
    5) 모델 버전 관리, 성능 모니터링, 감사 로깅
    6) 결과 CSV 저장

    :param cfg: 설정 딕셔너리
    :return: 이상 탐지 결과 DataFrame (anomaly 컬럼 포함)
    """
    # 1) 데이터 로드
    end = datetime.utcnow()
    lookback = cfg.get('lookback_days', 1)
    start = end - timedelta(days=lookback)
    df_raw = DBClient(cfg['db_uri']).fetch(start, end)
    if df_raw.empty:
        logging.warning("No activity logs for the given window")
        return pd.DataFrame()

    # 2) 피처 추출
    fe = FeatureExtractor(resource_weights=cfg.get('resource_weights', {}),
                          work_hours=(cfg.get('work_hours_start'), cfg.get('work_hours_end')))
    X = fe.transform(df_raw)

    # 3) 모델 학습/로드 및 탐지
    det = RiskAnomalyDetector(
        model_path=cfg['model_path'],
        contamination=cfg['contamination'],
        n_estimators=cfg.get('n_estimators', 100),
        random_state=cfg.get('random_state', 42)
    )
    det.load_or_train(X)
    results = det.detect(X)

    # 4) 알림 전처리 및 전송
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

    # 5) 모델 버전 관리 & 성능 모니터링
    # (a) 모델 버전 관리
    mlflow_uri = cfg.get('mlflow_tracking_uri')
    if mlflow_uri:
        registry = ModelRegistry(mlflow_uri, cfg.get('mlflow_experiment', 'AnomalyDetection'))
        run_id = registry.log_training(
            model=det.model,
            params={'contamination': cfg['contamination']},
            metrics={'num_samples': len(X), 'num_anomalies': int(results['anomaly'].sum())}
        )
        logging.info(f"Logged model run {run_id} to MLflow")
    # (b) 성능 모니터링
    if cfg.get('performance_thresholds'):
        monitor = PerformanceMonitor(cfg.get('performance_thresholds'))
        # 실제 레이블이 있으면 평가 가능
        if 'is_risk' in df_raw.columns:
            y_true = df_raw.groupby('user_id')['is_risk'].max().reindex(X.index).fillna(0).astype(int)
            p_metrics, p_alerts = monitor.evaluate(y_true, results['anomaly'])
            for alert in p_alerts:
                logging.warning(f"Performance alert: {alert}")

    # 6) 감사 추적 로깅
    if cfg.get('audit_db_uri'):
        auditor = AuditLogger(cfg['audit_db_uri'])
        auditor.log_event('detection', {
            'timestamp': datetime.utcnow().isoformat(),
            'num_anomalies': int(results['anomaly'].sum()),
            'params': {'contamination': cfg['contamination']}
        })

    # 7) 결과 저장
    out_csv = cfg.get('output_csv', 'anomaly_results.csv')
    results.to_csv(out_csv)
    logging.info(f"Results saved to {out_csv}")
    return results


if __name__ == '__main__':
    cfg = load_config('config.yaml')
    run_pipeline(cfg)