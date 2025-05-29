"""
SequenceAnomalyDetector
Word2Vec 임베딩 학습
LSTM Autoencoder 훈련 및 재구성 오차 기반 이상 시퀀스 판정
train(df) / detect(df) API 제공
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SequenceAnomalyDetector:
    """
    Sequence-based 이상탐지기 (Word2Vec + LSTM Autoencoder)

    - train(sequences): Word2Vec 임베딩 학습 및 LSTM 오토인코더 훈련
    - detect(sequences): 재구성 오차 기반 이상 시퀀스 판정
    """
    def __init__(
        self,
        embed_size: int = 50,
        lstm_units: int = 32,
        max_seq_len: int = 100,
        epochs: int = 10,
        batch_size: int = 32,
        contamination: float = 0.01
    ):
        self.embed_size = embed_size
        self.lstm_units = lstm_units
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.contamination = contamination
        self.w2v = None
        self.autoencoder = None
        self.threshold = None

    def _build_autoencoder(self):
        inputs = Input(shape=(self.max_seq_len, self.embed_size))
        x = Masking(mask_value=0.0)(inputs)
        encoded = LSTM(self.lstm_units)(x)
        decoded = RepeatVector(self.max_seq_len)(encoded)
        decoded = LSTM(self.embed_size, return_sequences=True)(decoded)
        model = Model(inputs, decoded)
        model.compile(optimizer='adam', loss='mse')
        return model

    def _prepare_sequences(self, df: pd.DataFrame) -> (list, np.ndarray):
        # 시퀀스 추출: 사용자별 action_type 리스트
        df = df.sort_values(['user_id', 'login_time'])
        grouped = df.groupby('user_id')['action_type'].apply(list)
        user_ids = grouped.index.tolist()
        sequences = grouped.tolist()
        # Word2Vec 임베딩 배열로 변환
        embedded = [ [self.w2v.wv[token] for token in seq] for seq in sequences ]
        # 패딩/트렁케이트
        X = pad_sequences(
            embedded,
            maxlen=self.max_seq_len,
            dtype='float32',
            padding='post',
            truncating='post'
        )
        return user_ids, X

    def train(self, df: pd.DataFrame):
        """
        Word2Vec + LSTM Autoencoder 학습
        :param df: user_id, login_time, action_type 컬럼 포함 데이터
        """
        # 1) Word2Vec 학습
        df['login_time'] = pd.to_datetime(df['login_time'])
        sequences = df.sort_values(['user_id','login_time']).groupby('user_id')['action_type'].apply(list).tolist()
        self.w2v = Word2Vec(sentences=sequences, vector_size=self.embed_size, window=5, min_count=1, epochs=10)
        # 2) 시퀀스 임베딩 및 패딩
        _, X = self._prepare_sequences(df)
        # 3) LSTM Autoencoder
        self.autoencoder = self._build_autoencoder()
        self.autoencoder.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=1
        )
        # 4) 재구성 오차 계산 및 임계값 설정
        X_pred = self.autoencoder.predict(X)
        mse = np.mean(np.square(X - X_pred), axis=(1,2))
        # contamination 비율에 따른 threshold
        self.threshold = np.quantile(mse, 1 - self.contamination)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이상 시퀀스 탐지
        :param df: user_id, login_time, action_type 컬럼 포함 데이터
        :return: DataFrame with columns [user_id, reconstruction_error, is_anomaly]
        """
        df['login_time'] = pd.to_datetime(df['login_time'])
        user_ids, X = self._prepare_sequences(df)
        X_pred = self.autoencoder.predict(X)
        mse = np.mean(np.square(X - X_pred), axis=(1,2))
        is_anomaly = (mse > self.threshold).astype(int)
        return pd.DataFrame({
            'user_id': user_ids,
            'reconstruction_error': mse,
            'is_anomaly': is_anomaly
        }).set_index('user_id')

# Example usage
if __name__ == '__main__':
    from db import DBClient
    from config import load_config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 데이터 로드
    end = pd.to_datetime('now')
    start = end - pd.Timedelta(days=cfg['lookback_days'])
    df = DBClient(cfg['db_uri']).fetch(start, end)

    # 모델 학습 및 이상 탐지
    seq_detector = SequenceAnomalyDetector(
        embed_size=cfg.get('embed_size', 50),
        lstm_units=cfg.get('lstm_units', 32),
        max_seq_len=cfg.get('max_seq_len', 100),
        contamination=cfg.get('contamination', 0.01)
    )
    seq_detector.train(df)
    results = seq_detector.detect(df)
    print(results.head())
