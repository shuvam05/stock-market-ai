from pathlib import Path
from datetime import datetime
from io import BytesIO
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors


# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "stock_model.pkl"
YFINANCE_CACHE_DIR = BASE_DIR / ".yfinance_cache"

YFINANCE_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


def flatten_columns(dataframe):
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)
    return dataframe


def normalize_date_column(dataframe):
    source_col = "Datetime" if "Datetime" in dataframe.columns else "Date"
    dates = pd.to_datetime(dataframe[source_col], errors="coerce")

    try:
        dates = dates.dt.tz_localize(None)
    except TypeError:
        pass

    dataframe["Date"] = dates
    return dataframe


def download_market_index(symbol, close_col, return_col, period, interval):
    index_df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
    )

    if index_df.empty:
        return pd.DataFrame(columns=["Date", close_col, return_col])

    index_df = flatten_columns(index_df)
    index_df.reset_index(inplace=True)
    index_df = normalize_date_column(index_df)

    index_df[close_col] = pd.to_numeric(index_df["Close"], errors="coerce")
    index_df[return_col] = index_df[close_col].pct_change()

    return index_df[["Date", close_col, return_col]]


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Stock Market AI Analyst",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div style='text-align:center;
padding:1rem;
background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
border-radius:14px;
margin-bottom:20px;'>

<h1 style='color:white;'>📈 Stock Market AI Analyst</h1>
<p style='color:#ccccff;'>Advanced AI Trend Classification Dashboard</p>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if hasattr(model, "n_jobs"):
    model.n_jobs = 1

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Market Controls")

ticker = st.sidebar.text_input("Search Company", "RELIANCE.NS")

interval_label = st.sidebar.selectbox(
    "Select Interval",
    ["1 Minute", "15 Minutes", "1 Hour", "1 Day"],
)

interval_map = {
    "1 Minute": "1m",
    "15 Minutes": "15m",
    "1 Hour": "60m",
    "1 Day": "1d",
}

interval = interval_map[interval_label]

if st.sidebar.button("🔄 Refresh Chart Now"):
    st.rerun()

live_refresh = st.sidebar.checkbox("Live Refresh (30 sec)", False)

if live_refresh:
    st.sidebar.success("Live mode active")
    st_autorefresh(interval=30000, key="live_refresh")

# -----------------------------
# MARKET STATUS
# -----------------------------
now = datetime.now()

market_open = (
    (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
    and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
)

market_status = "🟢 OPEN" if market_open else "🔴 CLOSED"
st.sidebar.markdown(f"### Market Status: {market_status}")

# -----------------------------
# CANDLE TIMER
# -----------------------------
interval_seconds = {
    "1m": 60,
    "15m": 900,
    "60m": 3600,
    "1d": 86400,
}

seconds = interval_seconds[interval]
remaining = seconds - (now.timestamp() % seconds)

mins = int(remaining // 60)
secs = int(remaining % 60)

st.sidebar.info(f"⏱ Next candle in: {mins:02d}:{secs:02d}")

# -----------------------------
# PERIOD
# -----------------------------
period = "7d" if interval in ["1m", "15m", "60m"] else "1y"

# -----------------------------
# FETCH STOCK
# -----------------------------
df = yf.download(ticker, period=period, interval=interval, progress=False)

if df.empty:
    st.error("No data found.")
    st.stop()

df = flatten_columns(df)
df.reset_index(inplace=True)
df = normalize_date_column(df)

# -----------------------------
# NIFTY + BANKNIFTY
# -----------------------------
nifty = download_market_index("^NSEI", "NIFTY_Close", "NIFTY_Return", period, interval)
banknifty = download_market_index(
    "^NSEBANK",
    "BANKNIFTY_Close",
    "BANKNIFTY_Return",
    period,
    interval,
)

df = pd.merge(df, nifty, on="Date", how="left")
df = pd.merge(df, banknifty, on="Date", how="left")

# -----------------------------
# INDICATORS
# -----------------------------
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
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

df["Close_Lag1"] = df["Close"].shift(1)
df["Close_Lag2"] = df["Close"].shift(2)
df["Close_Lag3"] = df["Close"].shift(3)

df.dropna(inplace=True)

if df.empty:
    st.error("Not enough valid data to calculate indicators.")
    st.stop()

# -----------------------------
# CHART LIMIT
# -----------------------------
chart_df = df.tail(200).copy() if interval == "1m" else df.copy()

# -----------------------------
# PREDICTION
# -----------------------------
latest = df.iloc[-1]

missing_features = [
    col for col in model.feature_names_in_
    if col not in df.columns
]

if missing_features:
    st.error(f"Model expects missing features: {', '.join(missing_features)}")
    st.stop()

latest_features = df[list(model.feature_names_in_)].tail(1)
prediction = model.predict(latest_features)[0]
action = "BUY 🟢" if prediction == 1 else "SELL 🔴"

# -----------------------------
# SIGNALS
# -----------------------------
chart_df["Signal"] = 0
chart_df.loc[chart_df["SMA_10"] > chart_df["SMA_50"], "Signal"] = 1
chart_df.loc[chart_df["SMA_10"] < chart_df["SMA_50"], "Signal"] = -1

buy = chart_df[chart_df["Signal"] == 1]
sell = chart_df[chart_df["Signal"] == -1]


# -----------------------------
# PDF REPORT FUNCTION
# -----------------------------
def create_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    data = [
        ["Field", "Value"],
        ["Company", ticker],
        ["Current Price", f"{latest['Close']:.2f}"],
        ["Volume", f"{int(latest['Volume'])}"],
        ["Suggested Action", action],
        ["Market Status", market_status],
        ["Generated At", now.strftime("%d-%m-%Y %H:%M:%S")],
    ]

    table = Table(data)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))

    doc.build([table])
    buffer.seek(0)
    return buffer


# -----------------------------
# METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

price_color = "#00E676" if latest["Close"] >= latest["Open"] else "#FF1744"

col1.markdown(f"""
<div style='font-size:28px;
font-weight:bold;
color:{price_color};
animation: blinker 1s linear infinite;'>
₹ {latest['Close']:.2f}
</div>

<style>
@keyframes blinker {{
  50% {{ opacity: 0.3; }}
}}
</style>
""", unsafe_allow_html=True)

col2.metric("Volume", int(latest["Volume"]))
col3.metric("Action", action)

# -----------------------------
# PDF DOWNLOAD
# -----------------------------
pdf_file = create_pdf()

st.download_button(
    label="📄 Download PDF Report",
    data=pdf_file,
    file_name="stock_report.pdf",
    mime="application/pdf",
)

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
    decreasing_line_color="#FF1744",
))

fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["SMA_10"],
    name="SMA 10",
))

fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["SMA_50"],
    name="SMA 50",
))

fig.add_trace(go.Scatter(
    x=buy["Date"],
    y=buy["Close"],
    mode="markers",
    marker=dict(symbol="triangle-up", size=12, color="#00E676"),
    name="Buy",
))

fig.add_trace(go.Scatter(
    x=sell["Date"],
    y=sell["Close"],
    mode="markers",
    marker=dict(symbol="triangle-down", size=12, color="#FF1744"),
    name="Sell",
))

latest_price = chart_df["Close"].iloc[-1]

fig.add_hline(y=latest_price, line_dash="dot", line_color=price_color)

fig.add_annotation(
    x=chart_df["Date"].iloc[-1],
    y=latest_price,
    text=f"{latest_price:.2f}",
    showarrow=False,
    bgcolor=price_color,
    font=dict(color="white"),
)

fig.update_layout(
    height=750,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
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
    name="RSI",
))

rsi_fig.add_hline(y=70)
rsi_fig.add_hline(y=30)

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
    name="MACD",
))

macd_fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["Signal_Line"],
    name="Signal Line",
))

macd_fig.update_layout(height=250, template="plotly_dark")

st.plotly_chart(macd_fig, use_container_width=True)

# -----------------------------
# DATA TABLE
# -----------------------------
st.subheader("Latest Data")
st.dataframe(chart_df.tail(20))
