import logging
import joblib
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap

class ClassificationModel:
    """
    분류(Classification) 모델 도입을 위한 트레이너 및 예측기
    - train(): 레이블링된 특성/타겟 데이터를 이용해 분류 모델 학습
    - predict(): 새로운 데이터에 대해 예측 라벨 및 확률 반환
    - explain(): SHAP 기반 설명 정보 생성
    """
    def __init__(
        self,
        model_path: str = 'classifier_model.joblib',
        contamination_model_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        self.model_path = model_path
        self.contamination_model_path = contamination_model_path
        self.params = params or {}
        self.model: Optional[RandomForestClassifier] = None
        self.explainer: Optional[Any] = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        분류 모델 학습 및 저장
        :param X: 특성 데이터프레임
        :param y: 레이블 시리즈
        """
        # 학습/검증 분리
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info('Training RandomForestClassifier...')
        model = RandomForestClassifier(
            n_estimators=self.params.get('n_estimators', 100),
            max_depth=self.params.get('max_depth', None),
            random_state=self.params.get('random_state', 42),
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        # 검증
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:,1]
        logging.info('Validation AUC: %.4f', roc_auc_score(y_val, y_proba))
        logging.info('Classification Report:\n%s', classification_report(y_val, y_pred))
        # 모델 저장
        joblib.dump(model, self.model_path)
        logging.info('Classification model saved to %s', self.model_path)
        self.model = model

        # SHAP explainer 준비
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception as e:
            logging.warning('SHAP TreeExplainer initialization failed: %s', e)

    def load(self):
        """저장된 모델 로드"""
        self.model = joblib.load(self.model_path)
        logging.info('Loaded classification model from %s', self.model_path)
        # explainer 초기화
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        새로운 데이터에 대해 예측 수행
        :param X: 특성 데이터프레임
        :return: DataFrame with columns ['prediction', 'probability']
        """
        if self.model is None:
            raise RuntimeError('Model not loaded. Call load() or train() first.')
        preds = self.model.predict(X)
        probas = self.model.predict_proba(X)[:,1]
        return pd.DataFrame({
            'prediction': preds,
            'probability': probas
        }, index=X.index)

    def explain(self, X: pd.DataFrame, max_display: int = 10) -> pd.DataFrame:
        """
        SHAP 값을 통해 피처 기여도 해석
        :param X: 특성 데이터프레임
        :param max_display: 요약 시각화 항목 수
        :return: SHAP 값 DataFrame (index): feature contributions
        """
        if self.explainer is None:
            raise RuntimeError('SHAP explainer not initialized.')
        shap_values = self.explainer.shap_values(X)
        # shap_values[1]는 Positive class 기여도
        df_shap = pd.DataFrame(shap_values[1], columns=X.columns, index=X.index)
        return df_shap.iloc[:, :max_display]


if __name__ == '__main__':
    import argparse
    from config import load_config
    from db import DBClient
    from features import FeatureExtractor
    
    parser = argparse.ArgumentParser(description='Train and evaluate classification model')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 데이터 로드 및 피처
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=cfg.get('lookback_days', 1))
    raw = DBClient(cfg['db_uri']).fetch(start, end)
    fe = FeatureExtractor(resource_weights=cfg.get('resource_weights', {}))
    X = fe.transform(raw)
    # 레이블 로드: audit 결과가 담긴 테이블/CSV
    # 예시: raw에 'is_risk' 컬럼이 있다고 가정
    y = raw.groupby('user_id').agg(is_risk=('is_risk','max'))
    y = y.reindex(X.index).fillna(0).astype(int)

    clf = ClassificationModel(model_path=cfg.get('classifier_model_path','classifier_model.joblib'),
                              params=cfg.get('classifier_params', {}))
    clf.train(X, y)
    preds = clf.predict(X)
    print(preds.head())
    if clf.explainer:
        shap_df = clf.explain(X)
        print(shap_df.head())
