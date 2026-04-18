from pathlib import Path
from datetime import datetime
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Stock Market AI Analyst",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "stock_model.pkl"

# -----------------------------
# LOAD MODEL
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if hasattr(model, "n_jobs"):
    model.n_jobs = 1

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div style='text-align:center;padding:1rem;
background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
border-radius:14px;margin-bottom:20px;'>

<h1 style='color:white;'>📈 Stock Market AI Analyst</h1>
<p style='color:#ccccff;'>Professional AI Trading Dashboard</p>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Market Controls")

ticker = st.sidebar.text_input("Search Company", "RELIANCE.NS")

interval_label = st.sidebar.selectbox(
    "Select Interval",
    ["1 Minute", "15 Minutes", "1 Hour", "1 Day"]
)

interval_map = {
    "1 Minute": "1m",
    "15 Minutes": "15m",
    "1 Hour": "60m",
    "1 Day": "1d"
}

interval = interval_map[interval_label]

if st.sidebar.button("🔄 Refresh Chart Now"):
    st.rerun()

live_refresh = st.sidebar.checkbox("Live Refresh (30 sec)")

if live_refresh:
    st_autorefresh(interval=30000, key="live")

# -----------------------------
# MARKET STATUS
# -----------------------------
now = datetime.now()

market_open = (
    (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
    and
    (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
)

status = "🟢 OPEN" if market_open else "🔴 CLOSED"
st.sidebar.markdown(f"### Market Status: {status}")

# -----------------------------
# COUNTDOWN TIMER
# -----------------------------
interval_seconds = {
    "1m": 60,
    "15m": 900,
    "60m": 3600,
    "1d": 86400
}

remaining = interval_seconds[interval] - (now.timestamp() % interval_seconds[interval])

mins = int(remaining // 60)
secs = int(remaining % 60)

st.sidebar.info(f"⏱ Next candle in: {mins:02d}:{secs:02d}")

# -----------------------------
# PERIOD
# -----------------------------
period = "7d" if interval in ["1m", "15m", "60m"] else "1y"

# -----------------------------
# STOCK DATA
# -----------------------------
df = yf.download(
    ticker,
    period=period,
    interval=interval,
    progress=False
)

if df.empty:
    st.error("No stock data found")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.reset_index(inplace=True)

if "Datetime" in df.columns:
    df["Date"] = pd.to_datetime(df["Datetime"])
else:
    df["Date"] = pd.to_datetime(df["Date"])

# -----------------------------
# NIFTY DATA
# -----------------------------
nifty = yf.download("^NSEI", period=period, interval=interval, progress=False)

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)

nifty.reset_index(inplace=True)

if "Datetime" in nifty.columns:
    nifty["Date"] = pd.to_datetime(nifty["Datetime"])
else:
    nifty["Date"] = pd.to_datetime(nifty["Date"])

nifty["NIFTY_Close"] = nifty["Close"]
nifty["NIFTY_Return"] = nifty["Close"].pct_change()

nifty = nifty[["Date", "NIFTY_Close", "NIFTY_Return"]]

# -----------------------------
# BANKNIFTY DATA
# -----------------------------
banknifty = yf.download("^NSEBANK", period=period, interval=interval, progress=False)

if isinstance(banknifty.columns, pd.MultiIndex):
    banknifty.columns = banknifty.columns.get_level_values(0)

banknifty.reset_index(inplace=True)

if "Datetime" in banknifty.columns:
    banknifty["Date"] = pd.to_datetime(banknifty["Datetime"])
else:
    banknifty["Date"] = pd.to_datetime(banknifty["Date"])

banknifty["BANKNIFTY_Close"] = banknifty["Close"]
banknifty["BANKNIFTY_Return"] = banknifty["Close"].pct_change()

banknifty = banknifty[["Date", "BANKNIFTY_Close", "BANKNIFTY_Return"]]

# -----------------------------
# MERGE
# -----------------------------
df = pd.merge(df, nifty, on="Date", how="left")
df = pd.merge(df, banknifty, on="Date", how="left")

df.ffill(inplace=True)
df.fillna(0, inplace=True)

# -----------------------------
# NUMERIC
# -----------------------------
for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# INDICATORS
# -----------------------------
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()

df["Daily_Return"] = df["Close"].pct_change()
df["Volatility"] = df["Daily_Return"].rolling(10).std()

# RSI
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = df["Close"].ewm(span=12).mean()
ema26 = df["Close"].ewm(span=26).mean()
df["MACD"] = ema12 - ema26
df["Signal_Line"] = df["MACD"].ewm(span=9).mean()

# Lag Features
df["Close_Lag1"] = df["Close"].shift(1)
df["Close_Lag2"] = df["Close"].shift(2)
df["Close_Lag3"] = df["Close"].shift(3)

df.dropna(inplace=True)

# -----------------------------
# CHART LIMIT
# -----------------------------
chart_df = df.tail(120).copy() if interval == "1m" else df.tail(250).copy()

# -----------------------------
# SAFE FEATURE MATCHING
# -----------------------------
required_features = list(model.feature_names_in_)

for col in required_features:
    if col not in chart_df.columns:
        chart_df[col] = 0

latest_features = chart_df[required_features].tail(1)

prediction = model.predict(latest_features)[0]

# -----------------------------
# ACTION
# -----------------------------
action = "BUY 🟢" if prediction == 1 else "SELL 🔴"

latest = chart_df.iloc[-1]

# -----------------------------
# SIGNALS (REAL CROSSOVER ONLY)
# -----------------------------
chart_df["Signal"] = 0
chart_df.loc[chart_df["SMA_10"] > chart_df["SMA_50"], "Signal"] = 1
chart_df.loc[chart_df["SMA_10"] < chart_df["SMA_50"], "Signal"] = -1

chart_df["Signal_Change"] = chart_df["Signal"].diff()

buy = chart_df[chart_df["Signal_Change"] == 2]
sell = chart_df[chart_df["Signal_Change"] == -2]

# -----------------------------
# METRICS
# -----------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Current Price", f"{latest['Close']:.2f}")
c2.metric("Volume", int(latest["Volume"]))
c3.metric("Suggested Action", action)

# -----------------------------
# MAIN CHART
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=chart_df["Date"],
    open=chart_df["Open"],
    high=chart_df["High"],
    low=chart_df["Low"],
    close=chart_df["Close"],
    increasing_line_color="#00C853",
    decreasing_line_color="#FF1744"
))

fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["SMA_10"],
    line=dict(color="#FF5722"),
    name="SMA 10"
))

fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["SMA_50"],
    line=dict(color="#00E5FF"),
    name="SMA 50"
))

fig.add_trace(go.Scatter(
    x=buy["Date"],
    y=buy["Close"],
    mode="markers",
    marker=dict(symbol="triangle-up", size=12, color="lime"),
    name="Buy"
))

fig.add_trace(go.Scatter(
    x=sell["Date"],
    y=sell["Close"],
    mode="markers",
    marker=dict(symbol="triangle-down", size=12, color="red"),
    name="Sell"
))

latest_price = chart_df["Close"].iloc[-1]

fig.add_hline(y=latest_price, line_dash="dot", line_color="yellow")

fig.update_layout(
    height=750,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# RSI
# -----------------------------
st.subheader("RSI")

rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["RSI"],
    line=dict(color="#00B0FF")
))

rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")

rsi_fig.update_layout(height=250, template="plotly_dark")

st.plotly_chart(rsi_fig, use_container_width=True)

# -----------------------------
# MACD
# -----------------------------
st.subheader("MACD")

macd_fig = go.Figure()

macd_fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["MACD"],
    line=dict(color="#FF9800")
))

macd_fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["Signal_Line"],
    line=dict(color="#00E676")
))

macd_fig.update_layout(height=250, template="plotly_dark")

st.plotly_chart(macd_fig, use_container_width=True)