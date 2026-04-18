from pathlib import Path
import pickle
import sys

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "stock_model.pkl"
YFINANCE_CACHE_DIR = BASE_DIR / ".yfinance_cache"

YFINANCE_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COMPANY_MAP = {
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "itc": "ITC.NS",
    "kotak": "KOTAKBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "wipro": "WIPRO.NS",
    "airtel": "BHARTIARTL.NS",
    "bharti": "BHARTIARTL.NS",
    "lt": "LT.NS",
    "l&t": "LT.NS",
    "ongc": "ONGC.NS",
    "tatamotors": "TATAMOTORS.NS",
    "tata motors": "TATAMOTORS.NS",
    "maruti": "MARUTI.NS",
    "hindunilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",
}


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def normalize_dates(df):
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def resolve_ticker(company):
    query = company.lower().strip()

    if not query:
        return None

    if query.endswith(".ns"):
        return query.upper()

    for name, ticker in COMPANY_MAP.items():
        if query in name:
            return ticker

    for name, ticker in COMPANY_MAP.items():
        if name in query:
            return ticker

    return None


def download_price_data(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")

    df = flatten_columns(df)
    df.reset_index(inplace=True)
    df = normalize_dates(df)

    return df


def download_index(symbol, close_col, return_col):
    df = download_price_data(symbol)

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

    return df


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    if len(sys.argv) > 1:
        company = " ".join(sys.argv[1:])
    else:
        company = input("Enter company name: ")

    ticker = resolve_ticker(company)

    if ticker is None:
        print("Company not found.")
        return

    print("Selected ticker:", ticker)

    df = download_price_data(ticker)

    nifty = download_index("^NSEI", "NIFTY_Close", "NIFTY_Return")
    banknifty = download_index("^NSEBANK", "BANKNIFTY_Close", "BANKNIFTY_Return")

    df = pd.merge(df, nifty, on="Date", how="left")
    df = pd.merge(df, banknifty, on="Date", how="left")

    df = add_features(df)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("Not enough data for prediction.")

    feature_cols = list(model.feature_names_in_)
    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        raise ValueError(f"Model expects missing features: {', '.join(missing_features)}")

    latest = df[feature_cols].tail(1)
    prediction = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]
    confidence = max(prob) * 100

    print("\n======================")

    if prediction == 1:
        print("Predicted Trend: UP 📈")
        print("Suggested Action: BUY 🟢")
    else:
        print("Predicted Trend: DOWN 📉")
        print("Suggested Action: SELL 🔴")

    print(f"Confidence: {confidence:.2f}%")

    vol = latest["Volatility"].iloc[0]
    avg_vol = df["Volatility"].mean()

    if vol < avg_vol:
        print("Risk Level: LOW 🟢")
    elif vol < avg_vol * 1.5:
        print("Risk Level: MEDIUM 🟡")
    else:
        print("Risk Level: HIGH 🔴")

    print("======================")


if __name__ == "__main__":
    main()
