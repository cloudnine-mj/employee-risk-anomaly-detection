# FeatureExtractor
# off-hours login ratio: 근무시간(9–18시) 외 로그인 비율(off_hours_login_ratio)
# weighted resource score: 리소스별 가중치 합(weighted_resource_score)
# 기본 피처(로그인 횟수, 리소스 종류 수, 거래 합계, 액션 타입 수) 유지

import pandas as pd

class FeatureExtractor:
    WORK_START_HOUR = 9
    WORK_END_HOUR   = 18

    def __init__(self, resource_weights: dict = None):
        self.resource_weights = resource_weights or {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 로그인 시간 datetime
        df['login_time'] = pd.to_datetime(df['login_time'])
        # 시간 정보 및 야간/주말 로그인 플래그 (off_hours)
        df['hour'] = df['login_time'].dt.hour
        df['off_hours'] = df['hour'].apply(
            lambda h: 1 if (h < self.WORK_START_HOUR or h >= self.WORK_END_HOUR) else 0
        )
        # 리소스 가중치 매핑 (기본 1.0)
        df['res_weight'] = df['resource_id'].map(self.resource_weights).fillna(1.0)

        # 그룹별 집계
        features = df.groupby('user_id').agg(
            login_count             = ('login_time', 'count'),
            resource_count          = ('resource_id','nunique'),
            total_amount            = ('amount','sum'),
            action_type_count       = ('action_type','nunique'),
            off_hours_login_ratio   = ('off_hours','mean'),
            weighted_resource_score = ('res_weight','sum')
        )

        # 결측값 0으로 대체
        features = features.fillna(0)
        return features


if __name__ == '__main__':
    import argparse
    from datetime import datetime, timedelta
    from data.db import DBClient
    from config import load_config

    parser = argparse.ArgumentParser(description='Generate feature vectors')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    end = datetime.utcnow()
    start = end - timedelta(days=cfg.get('lookback_days',1))
    raw = DBClient(cfg['db_uri']).fetch(start, end)
    fe = FeatureExtractor(resource_weights=cfg.get('resource_weights', {}))
    feats = fe.transform(raw)
    print(feats.head())



