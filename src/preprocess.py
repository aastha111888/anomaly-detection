from datetime import date

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    "daily_return",
    "rolling_volatility",
    "volume_change",
    "high_low_spread",
]


def load_and_preprocess():
    raw = yf.download(
        "SPY",
        start="2005-01-01",
        end=date.today().strftime("%Y-%m-%d"),
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["rolling_volatility"] = df["daily_return"].rolling(window=7).std()
    df["volume_change"] = df["Volume"].pct_change()
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]

    df = df.dropna()

    train_mask = df.index < "2020-01-01"
    test_mask = df.index >= "2020-01-01"

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df.loc[train_mask, FEATURE_COLS])
    X_test = scaler.transform(df.loc[test_mask, FEATURE_COLS])

    return X_train, X_test, df, scaler


if __name__ == "__main__":
    X_train, X_test, df, scaler = load_and_preprocess()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("\nFirst 5 rows of df:")
    print(df.head())
