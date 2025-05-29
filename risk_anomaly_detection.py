import argparse
import json
import logging
import sys
from datetime import datetime, timedelta

import joblib
import pandas as pd
import sqlalchemy
import yaml
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from river import drift
from sklearn.ensemble import IsolationForest
from alerting.alertmanager import AlertmanagerClient
from alerting.suppression import filter_alerts, FlappingSuppressor, Deduplicator, MuteList
from retrain_scheduler import RetrainScheduler

# 설정 로더
class ConfigLoader:
    @staticmethod
    def load(path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

# 로깅 셋업 (JSON 포맷)
def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(json.dumps({
        "timestamp": "%(asctime)s",
        "level": "%(levelname)s",
        "module": "%(module)s",
        "message": "%(message)s"
    })))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]

# Prometheus 메트릭 정의 
REQUEST_LATENCY = Histogram('risk_detection_latency_seconds', 'Detection latency')
ERROR_COUNT     = Counter('risk_detection_errors_total', 'Errors during detection')
ANOMALIES_TOTAL = Counter('risk_detection_anomalies_total', 'Total anomalies detected')
LAST_RUN        = Gauge('risk_detection_last_run_timestamp', 'Last run timestamp')

def start_metrics_server(port: int):
    start_http_server(port)
    logging.info(f"Metrics server listening on :{port}")

def instrumented(func):
    def wrapper(*args, **kwargs):
        start = datetime.utcnow()
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            ERROR_COUNT.inc()
            logging.exception("Detection error")
            raise
        duration = (datetime.utcnow() - start).total_seconds()
        REQUEST_LATENCY.observe(duration)
        LAST_RUN.set_to_current_time()
        if hasattr(res, 'anomaly'):
            ANOMALIES_TOTAL.inc(int(res.anomaly.sum()))
        return res
    return wrapper

# DB 클라이언트 
class DBClient:
    def __init__(self, uri: str):
        self.engine = sqlalchemy.create_engine(uri)
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        sql = """SELECT user_id, login_time, action_type, resource_id, amount
                 FROM employee_activity_logs
                 WHERE login_time BETWEEN :start AND :end"""
        return pd.read_sql(sql, self.engine, params={'start': start, 'end': end})

# 피처 변환
class FeatureExtractor:
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df['login_time'] = pd.to_datetime(df['login_time'])
        df = df.fillna({'amount': 0})
        agg = df.groupby('user_id').agg(
            login_count       = ('login_time','count'),
            resource_count    = ('resource_id','nunique'),
            total_amount      = ('amount','sum'),
            action_type_count = ('action_type','nunique')
        )
        return agg

# 이상탐지기 
class RiskAnomalyDetector:
    def __init__(self, cfg):
        self.model_path    = cfg['model_path']
        self.contamination = cfg['contamination']
        self.model         = None
        self.adwin         = drift.ADWIN() if cfg.get('drift_detection') else None

    def load_or_train(self, X: pd.DataFrame):
        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Loaded model from {self.model_path}")
        except FileNotFoundError:
            logging.info("Training new IsolationForest")
            self.model = IsolationForest(contamination=self.contamination, n_jobs=-1, random_state=42)
            self.model.fit(X)
            joblib.dump(self.model, self.model_path)
            logging.info(f"Saved model to {self.model_path}")

    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        labels = self.model.predict(X)
        res = X.copy()
        res['anomaly'] = (labels == -1).astype(int)
        # 드리프트 감지
        if self.adwin:
            for val in X.sum(axis=1):
                if self.adwin.update(val):
                    logging.info("Data drift detected, retraining model")
                    self.model = None
                    break
        return res

# 메인 파이프라인 
@instrumented
def run_pipeline(cfg):
    end   = datetime.utcnow()
    start = end - timedelta(days=cfg['lookback_days'])
    # 1) 데이터 로드
    df   = DBClient(cfg['db_uri']).fetch(start, end)
    if df.empty:
        logging.warning("No data in window")
        return pd.DataFrame()
    # 2) 피처 변환
    feats = FeatureExtractor.transform(df)
    # 3) 모델 로드/학습 & 탐지
    det = RiskAnomalyDetector(cfg)
    det.load_or_train(feats)
    results = det.detect(feats)
    # 4) 알림 억제 후 전송
    alerts = results[results['anomaly']==1]
    alerts_payload = [
        {'labels': {'alertname':'RiskAnomaly','user_id':uid}, 'startsAt':datetime.utcnow().isoformat()+'Z'}
        for uid in alerts.index
    ]
    sup = FlappingSuppressor(timedelta(minutes=5), threshold=3)
    ded = Deduplicator(timedelta(minutes=10))
    mutes = MuteList([{'alertname':'RiskAnomaly','severity':'info'}])
    filtered = filter_alerts(alerts_payload, sup, ded, mutes)
    if cfg.get('alertmanager_url'):
        AlertmanagerClient(cfg['alertmanager_url']).send_alerts(filtered)
    # 5) 결과 저장
    results.to_csv(cfg['output_csv'])
    logging.info(f"Results saved to {cfg['output_csv']}")
    return results

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    cfg = ConfigLoader.load(args.config)
    # 메트릭 서버
    start_metrics_server(cfg.get('metrics_port', 8000))
    # 자동 재학습 스케줄러 (예: 하루 1회, 이벤트 5회·10분 내)
    if cfg.get('enable_retrain_scheduler'):
        def retrain(): run_pipeline(cfg)
        sched = RetrainScheduler(retrain,
                                 periodic_interval_hours=cfg.get('retrain_hourly',24),
                                 event_threshold=cfg.get('retrain_event_threshold',5),
                                 event_window_minutes=cfg.get('retrain_event_window',10))
    # 최초 실행
    run_pipeline(cfg)

if __name__ == '__main__':
    main()
