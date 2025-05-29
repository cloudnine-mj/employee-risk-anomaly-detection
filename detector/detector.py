#RiskAnomalyDetector 클래스
#load_or_train(): 모델 로드 또는 학습 후 joblib 저장
#detect(): 이상치 예측, anomaly 컬럼(1: 이상치, 0: 정상) 추가

import joblib
import logging
from typing import Optional
import pandas as pd
from sklearn.ensemble import IsolationForest


class RiskAnomalyDetector:
    """
    IsolationForest 기반 리스크 이상 탐지기
    - load_or_train(): 기존 모델 파일이 없으면 학습 후 저장
    - detect(): 입력 데이터에 대해 이상치 예측
    """
    def __init__(
        self,
        model_path: str,
        contamination: float = 0.02,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.model_path = model_path
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None

    def load_or_train(self, X: pd.DataFrame) -> None:
        """
        모델 파일이 존재하면 로드하고, 없으면 IsolationForest 학습 후 저장
        :param X: 학습 데이터 (n_samples, n_features)
        """
        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Loaded model from {self.model_path}")
        except (FileNotFoundError, EOFError):
            logging.info(
                f"Training IsolationForest (contamination={self.contamination}, "
                f"n_estimators={self.n_estimators})"
            )
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X)
            joblib.dump(self.model, self.model_path)
            logging.info(f"Model trained and saved to {self.model_path}")

    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        입력 데이터에 대해 이상치 여부를 예측하여 DataFrame으로 반환
        :param X: 탐지할 데이터 (n_samples, n_features)
        :return: 원본 X에 'anomaly' 컬럼(1 이상치, 0 정상) 추가된 DataFrame
        """
        if self.model is None:
            raise RuntimeError("Model not loaded or trained. Call load_or_train() first.")
        labels = self.model.predict(X)
        df = X.copy()
        df['anomaly'] = (labels == -1).astype(int)
        return df

if __name__ == '__main__':
    import argparse
    from config import load_config
    from data.db import DBClient
    from features.features import FeatureExtractor
    from pipeline import run_pipeline  # if pipeline exists

    parser = argparse.ArgumentParser(description='Train and detect with IsolationForest')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 데이터 로드 및 피처
    end = datetime.utcnow()
    start = end - timedelta(days=cfg.get('lookback_days', 1))
    raw = DBClient(cfg['db_uri']).fetch(start, end)
    feats = FeatureExtractor(cfg.get('resource_weights', {})).transform(raw)

    # 모델 학습/로드 및 탐지
    det = RiskAnomalyDetector(
        model_path=cfg['model_path'],
        contamination=cfg['contamination'],
        n_estimators=cfg.get('n_estimators', 100)
    )
    det.load_or_train(feats)
    results = det.detect(feats)
    print(results.head())
