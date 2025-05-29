import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SequenceAnomalyDetector:
    def __init__(self, embed_size=50, lstm_units=32, max_seq_len=100, epochs=10, batch_size=32):
        self.embed_size = embed_size
        self.lstm_units = lstm_units
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.w2v = None
        self.model = None

    def fit(self, action_seqs):
        tokenized = [seq for seq in action_seqs]
        self.w2v = Word2Vec(sentences=tokenized, vector_size=self.embed_size, window=5, min_count=1, workers=2)

        embedded_seqs = [
            [self.w2v.wv[token] for token in seq if token in self.w2v.wv]
            for seq in tokenized
        ]
        padded = pad_sequences(embedded_seqs, maxlen=self.max_seq_len, dtype='float32', padding='post', truncating='post')

        self.model = Sequential([
            Masking(mask_value=0.0, input_shape=(self.max_seq_len, self.embed_size)),
            LSTM(self.lstm_units),
            Dense(self.embed_size)
        ])
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(padded, padded[:, 0, :], epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def detect(self, action_seqs):
        embedded_seqs = [
            [self.w2v.wv[token] for token in seq if token in self.w2v.wv]
            for seq in action_seqs
        ]
        padded = pad_sequences(embedded_seqs, maxlen=self.max_seq_len, dtype='float32', padding='post', truncating='post')

        recon = self.model.predict(padded, verbose=0)
        errors = np.linalg.norm(padded[:, 0, :] - recon, axis=1)

        threshold = np.percentile(errors, 95)
        return pd.DataFrame({
            'sequence_anomaly_score': errors,
            'sequence_anomaly': (errors > threshold).astype(int)
        })