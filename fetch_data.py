from pathlib import Path
import pickle

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "stock_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.pkl"
COMBINED_DATA_PATH = DATA_DIR / "advanced_market_data.csv"
PREDICTIONS_PATH = DATA_DIR / "test_predictions.csv"
YFINANCE_CACHE_DIR = BASE_DIR / ".yfinance_cache"

YFINANCE_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "KOTAKBANK.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "BHARTIARTL.NS",
    "LT.NS",
    "WIPRO.NS",
    "ONGC.NS",
    "TATAMOTORS.NS",
    "MARUTI.NS",
]

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_50",
    "Daily_Return", "Volatility",
    "RSI", "MACD",
    "Close_Lag1", "Close_Lag2", "Close_Lag3",
    "NIFTY_Close", "NIFTY_Return",
    "BANKNIFTY_Close", "BANKNIFTY_Return",
]


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def normalize_dates(df):
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def download_index(symbol, close_col, return_col):
    df = yf.download(symbol, period="5y", interval="1d", progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df = flatten_columns(df)
    df.reset_index(inplace=True)
    df = normalize_dates(df)

    df[close_col] = pd.to_numeric(df["Close"], errors="coerce")
    df[return_col] = df[close_col].pct_change()

    return df[["Date", close_col, return_col]]


def add_features(df):
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(10).std()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Close_Lag3"] = df["Close"].shift(3)

    df["Future_Return"] = df["Close"].shift(-1) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > 0.005).astype(int)

    return df


def download_stock_data(stock, nifty, banknifty):
    print("Downloading:", stock)

    df = yf.download(stock, period="5y", interval="1d", progress=False)

    if df.empty:
        print("Skipped:", stock, "| Error: no data returned")
        return None

    df = flatten_columns(df)
    df.reset_index(inplace=True)
    df = normalize_dates(df)

    df = pd.merge(df, nifty, on="Date", how="left")
    df = pd.merge(df, banknifty, on="Date", how="left")
    df["Ticker"] = stock

    df = add_features(df)
    df.dropna(inplace=True)

    if df.empty:
        print("Skipped:", stock, "| Error: not enough valid rows after features")
        return None

    return df


def build_dataset():
    nifty = download_index("^NSEI", "NIFTY_Close", "NIFTY_Return")
    banknifty = download_index("^NSEBANK", "BANKNIFTY_Close", "BANKNIFTY_Return")

    all_data = []

    for stock in STOCKS:
        try:
            stock_df = download_stock_data(stock, nifty, banknifty)
            if stock_df is not None:
                all_data.append(stock_df)
        except Exception as exc:
            print("Skipped:", stock, "| Error:", exc)

    if not all_data:
        raise RuntimeError("No stock data could be downloaded.")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df.sort_values(["Date", "Ticker"], inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    return full_df


def train_model(full_df):
    print("Total rows:", len(full_df))

    X = full_df[FEATURE_COLS]
    y = full_df["Target"]

    split = int(len(full_df) * 0.8)

    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print("\nAccuracy:", round(acc * 100, 2), "%")
    print(classification_report(y_test, pred, zero_division=0))

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "feature_importances": dict(zip(FEATURE_COLS, model.feature_importances_)),
    }

    pred_df = X_test.copy()
    pred_df["Target"] = y_test.values
    pred_df["Predicted"] = pred

    return model, metrics, pred_df


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    full_df = build_dataset()
    full_df.to_csv(COMBINED_DATA_PATH, index=False)

    model, metrics, pred_df = train_model(full_df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)

    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    print("Advanced model saved successfully!")
    print("Metrics saved successfully!")
    print(f"Combined data saved to {COMBINED_DATA_PATH}")


if __name__ == "__main__":
    main()
