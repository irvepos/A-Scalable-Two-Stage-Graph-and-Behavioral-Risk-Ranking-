import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import entropy

class BehavioralScreeningFilter:
    def __init__(self, tau_screen=0.35, window_hours=24):
        self.tau_screen = tau_screen
        self.window_hours = window_hours
        self.beh_model = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1)
        self.is_trained = False

    def calculate_behavioral_features(self, df):
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        df['Velocity'] = df.groupby('source_account')['amount'].transform(
            lambda x: x.rolling(window=self.window_hours, min_periods=1).count()
        )
        
        fan_in = df.groupby('destination_account')['amount'].transform('sum')
        fan_out = df.groupby('source_account')['amount'].transform('sum')
        df['Flow_Skewness'] = fan_in / (fan_in + fan_out + 1e-7)
        
        def calc_entropy(amounts):
            counts, _ = np.histogram(amounts, bins=10)
            probs = counts / (counts.sum() + 1e-7)
            return entropy(probs, base=2)
            
        df['Amount_Entropy'] = df.groupby('source_account')['amount'].transform(
            lambda x: x.rolling(window=self.window_hours, min_periods=1).apply(calc_entropy, raw=True)
        ).fillna(0)
        
        return df

    def train_filter(self, df, features, target):
        X = df[features]
        y = df[target]
        self.beh_model.fit(X, y)
        self.is_trained = True

    def isolate_candidate_subgraphs(self, df, features):
        if not self.is_trained:
            raise ValueError("مدل فیلتر رفتاری ابتدا باید آموزش داده شود.")
            
        df['S_beh'] = self.beh_model.predict_proba(df[features])[:, 1]
        high_risk_candidates = df[df['S_beh'] >= self.tau_screen]
        
        pruning_ratio = 100 - (len(high_risk_candidates) / len(df) * 100)
        print(f"نرخ هرس تراکنش‌های امن (Pruning): {pruning_ratio:.2f}%")
        
        return high_risk_candidates