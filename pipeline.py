"""
run_pipeline(cfg):
- DB에서 지정 기간 로그 로드
- FeatureExtractor로 피처 생성
- RiskAnomalyDetector로 모델 학습/로딩 및 이상 탐지
- alerting.filter_alerts를 통해 Flapping/Dedup/Mute 적용 후 Alertmanager 전송
- 탐지 결과를 CSV로 저장
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
from config import load_config
from metrics import instrumented
from db import DBClient
from features import FeatureExtractor
from detector import RiskAnomalyDetector
from alerting import AlertmanagerClient, filter_alerts, FlappingSuppressor, Deduplicator, MuteList

@instrumented
def run_pipeline(cfg: dict) -> pd.DataFrame:
    """
    전체 리스크 탐지 파이프라인

    1) 데이터 로드
    2) 피처 추출
    3) 모델 학습/로딩 및 이상 탐지
    4) 알림 억제 로직 적용 후 Alertmanager로 전송
    5) 결과 CSV 저장

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
    fe = FeatureExtractor(resource_weights=cfg.get('resource_weights', {}))
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
         'annotations': {'summary': 'User risk anomaly'},
         'startsAt': datetime.utcnow().isoformat() + 'Z'}
        for uid in anomalies.index
    ]
    flapper = FlappingSuppressor(timedelta(minutes=cfg.get('flap_window_minutes', 5)),
                                 threshold=cfg.get('flap_threshold', 3))
    deduper = Deduplicator(timedelta(minutes=cfg.get('dedup_window_minutes', 10)))
    mutes = MuteList(cfg.get('mute_rules', []))
    filtered_alerts = filter_alerts(alerts, flapper, deduper, mutes)
    if cfg.get('alertmanager_url'):
        client = AlertmanagerClient(cfg['alertmanager_url'])
        client.send_alerts(filtered_alerts)

    # 5) 결과 저장
    out_csv = cfg.get('output_csv', 'anomaly_results.csv')
    results.to_csv(out_csv)
    logging.info(f"Results saved to {out_csv}")
    return results

if __name__ == '__main__':
    from config import load_config
    cfg = load_config('config.yaml')
    run_pipeline(cfg)
