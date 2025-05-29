import pandas as pd


class ClassificationFeatureGenerator:
 # 분류 모델 학습을 위한 피처 생성기

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        로그인 로그 등에서 분류용 피처를 추출
        :param df: 원본 로그 DataFrame
        :return: 사용자 단위로 Aggregation된 분류용 피처 DataFrame
        """
        features = df.groupby('user_id').agg(
            total_actions=('action_type', 'count'),
            unique_resources=('resource_id', 'nunique'),
            avg_interval_sec=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / len(x)),
            login_days=('timestamp', lambda x: x.dt.date.nunique())
        )
        return features
