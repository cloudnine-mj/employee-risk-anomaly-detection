
import pandas as pd
from datetime import datetime

class FeatureExtractor:
    def __init__(self, resource_weights: dict = None, work_hours=("09:00", "18:00")):
        self.resource_weights = resource_weights or {}
        self.start_hour, self.end_hour = [int(t.split(":")[0]) for t in work_hours]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 업무시간 외 로그인 비율 계산
        df['hour'] = pd.to_datetime(df['login_time']).dt.hour
        df['off_hours'] = ~df['hour'].between(self.start_hour, self.end_hour)
        ratio_off_hours = df.groupby('user_id')['off_hours'].mean().rename("off_hours_login_ratio")

        # 리소스 중요도 가중치 계산
        df['resource_weight'] = df['resource_id'].map(self.resource_weights).fillna(1.0)
        resource_score = df.groupby('user_id')['resource_weight'].sum().rename("weighted_resource_score")

        # 로그 수 (활동량)
        activity_count = df.groupby('user_id').size().rename("log_count")

        # 통합 피처
        features = pd.concat([ratio_off_hours, resource_score, activity_count], axis=1).fillna(0)
        return features
