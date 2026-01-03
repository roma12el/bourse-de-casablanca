
import numpy as np
def add_features(df):
    df['return'] = np.log(df['close']/df['close'].shift(1))
    df['volatility'] = df['return'].rolling(10).std()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df.dropna(inplace=True)
    return df
