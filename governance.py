import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score
from prometheus_client import Gauge, Counter
import sqlalchemy
import json
from datetime import datetime

# 모델 버전 관리 & 재현성
class ModelRegistry:
    """
    MLflow를 활용한 모델 버전 관리 및 재현성 보장
    """
    def __init__(self, tracking_uri: str, experiment_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_id = mlflow.create_experiment(experiment_name)

    def log_training(self, model, params: dict, metrics: dict, artifact_path: str = 'model'):
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            mlflow.log_params(params)
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            mlflow.sklearn.log_model(model, artifact_path)
        return run.info.run_id

    def load_model(self, run_id: str, artifact_path: str = 'model'):
        uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.sklearn.load_model(uri)

# 모델 성능 모니터링
# Prometheus 메트릭 정의
PRECISION_GAUGE = Gauge('model_precision', 'Classification precision')
RECALL_GAUGE = Gauge('model_recall', 'Classification recall')
F1_GAUGE = Gauge('model_f1', 'Classification F1 score')

class PerformanceMonitor:
    """
    Precision, Recall, F1 지표를 주기적으로 계산해 Prometheus로 노출
    """
    def __init__(self, alert_thresholds: dict = None):
        # alert_thresholds: {'precision':0.8, 'recall':0.7, 'f1':0.75}
        self.thresholds = alert_thresholds or {}
        # 알림 카운터
        self.alert_counter = Counter('model_performance_alerts_total', 'Number of performance alerts')

    def evaluate(self, y_true, y_pred):
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # 메트릭 갱신
        PRECISION_GAUGE.set(p)
        RECALL_GAUGE.set(r)
        F1_GAUGE.set(f1)
        # 임계치 이하 시 알림
        alerts = []
        for name, threshold in self.thresholds.items():
            value = locals()[name]
            if value < threshold:
                self.alert_counter.inc()
                alerts.append(f"{name} below threshold: {value:.2f} < {threshold}")
        return {'precision': p, 'recall': r, 'f1': f1}, alerts

# 감사 추적 로깅 
class AuditLogger:
    """
    탐지 및 알림 이벤트를 감사DB에 기록
    """
    def __init__(self, db_uri: str, table_name: str = 'audit_events'):
        self.engine = sqlalchemy.create_engine(db_uri)
        self.table = table_name
        # 테이블 생성 (간단 스키마)
        with self.engine.begin() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id SERIAL PRIMARY KEY,
                    event_time TIMESTAMP,
                    event_type VARCHAR(50),
                    details JSONB
                );
            """)

    def log_event(self, event_type: str, details: dict):
        """
        이벤트 기록
        :param event_type: 'detection' 또는 'alert'
        :param details: 이벤트 관련 추가 정보
        """
        now = datetime.utcnow()
        details_json = json.dumps(details)
        with self.engine.begin() as conn:
            conn.execute(f"INSERT INTO {self.table} (event_time, event_type, details) VALUES (:time, :type, :details)",
                         {'time': now, 'type': event_type, 'details': details_json})
