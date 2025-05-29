import argparse
import logging
import sys
from datetime import datetime, timedelta

import joblib
import pandas as pd
import sqlalchemy
from sklearn.ensemble import IsolationForest

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(module)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

class DBClient:
    """
    Database client for fetching employee activity logs.
    """
    def __init__(self, db_uri: str):
        self.engine = sqlalchemy.create_engine(db_uri)

    def fetch_activity(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = """
        SELECT
            user_id,
            login_time,
            action_type,
            resource_id,
            amount
        FROM employee_activity_logs
        WHERE login_time BETWEEN :start AND :end
        """
        return pd.read_sql(query, self.engine, params={
            'start': start,
            'end': end
        })

class FeatureExtractor:
    """
    Convert raw activity logs into feature vectors for anomaly detection.
    """
    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        # Ensure datetime
        df['login_time'] = pd.to_datetime(df['login_time'])
        # Feature: login count per user
        login_counts = df.groupby('user_id').login_time.count().rename('login_count')
        # Feature: distinct resources accessed
        resource_counts = df.groupby('user_id').resource_id.nunique().rename('resource_count')
        # Feature: total transaction amount
        total_amount = df.groupby('user_id').amount.sum().fillna(0).rename('total_amount')
        # Feature: unique action types
        action_types = df.groupby('user_id').action_type.nunique().rename('action_type_count')

        features = pd.concat([login_counts, resource_counts, total_amount, action_types], axis=1)
        features = features.fillna(0)
        return features

class RiskAnomalyDetector:
    """
    Trains or loads an IsolationForest model and detects anomalies in features.
    """
    def __init__(self,
                 model_path: str = 'risk_iforest.joblib',
                 contamination: float = 0.02):
        self.model_path = model_path
        self.contamination = contamination
        self.model = None

    def load_or_train(self, X: pd.DataFrame):
        try:
            self.model = joblib.load(self.model_path)
            logging.info("Loaded existing model from %s", self.model_path)
        except FileNotFoundError:
            logging.info("Training IsolationForest model with contamination=%.3f", self.contamination)
            self.model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X)
            joblib.dump(self.model, self.model_path)
            logging.info("Model saved to %s", self.model_path)

    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        labels = self.model.predict(X)
        results = X.copy()
        results['anomaly'] = (labels == -1).astype(int)
        return results

    def alert(self, anomalies: pd.DataFrame):
        if anomalies.empty:
            logging.info("No anomalies detected.")
            return
        logging.warning("Detected %d anomalous users", len(anomalies))
        for user_id, row in anomalies.iterrows():
            logging.warning(
                f"User {user_id} anomaly: login_count={row['login_count']}, "
                f"resource_count={row['resource_count']}, total_amount={row['total_amount']:.2f}, "
                f"action_types={row['action_type_count']}"
            )

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Employee risk anomaly detection for compliance monitoring"
    )
    parser.add_argument('--db-uri', required=True, help='SQLAlchemy database URI')
    parser.add_argument('--days', type=int, default=1, help='Lookback window in days')
    parser.add_argument('--contamination', type=float, default=0.02,
                        help='Expected fraction of anomalies')
    parser.add_argument('--model-path', default='risk_iforest.joblib', help='Path to model file')
    args = parser.parse_args()

    end = datetime.utcnow()
    start = end - timedelta(days=args.days)

    db = DBClient(args.db_uri)
    raw = db.fetch_activity(start, end)
    if raw.empty:
        logging.error("No activity logs found in the specified window.")
        sys.exit(1)

    features = FeatureExtractor.transform(raw)
    detector = RiskAnomalyDetector(
        model_path=args.model_path,
        contamination=args.contamination
    )
    detector.load_or_train(features)
    results = detector.detect(features)
    anomalies = results[results['anomaly'] == 1]
    detector.alert(anomalies)

    # Optionally, save results
    results.to_csv('anomaly_scores.csv')
    logging.info("Anomaly scores written to anomaly_scores.csv")

if __name__ == '__main__':
    main()