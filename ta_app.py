# ta_app.py
# -----------------------------------------
# TA Scout — Support/Resistance & Breakouts
# -----------------------------------------
# pip install streamlit yfinance pandas numpy plotly scipy

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from scipy.signal import argrelextrema

st.set_page_config(page_title="TA Scout", layout="wide")

# ============ Helpers for Yahoo constraints ============

# Yahoo intraday windows (typical)
_INTRADAY_MAX_PERIOD = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",  # yfinance maps 1h -> 60m; often 730d works
    "90m": "60d",
    "1h": "730d",
}

def is_intraday(interval: str) -> bool:
    return any(interval.endswith(suf) for suf in ("m", "h"))

def coerce_period_for_interval(interval: str, period: str) -> str:
    """If the user picks an intraday interval with too-long period, clamp it."""
    if not is_intraday(interval):
        return period
    maxp = _INTRADAY_MAX_PERIOD.get(interval, "60d")
    # Simple precedence: if period is longer than maxp, return maxp
    order = ["7d","14d","30d","45d","60d","90d","180d","1y","2y","5y","10y","max","730d"]
    # convert 730d position between 1y and 2y
    def pos(p):
        p = p.lower()
        if p.endswith("d"):
            try:
                d = int(p[:-1])
                # map roughly into order
                if d <= 7: return 0
                if d <= 14: return 1
                if d <= 30: return 2
                if d <= 45: return 3
                if d <= 60: return 4
                if d <= 90: return 5
                if d <= 180: return 6
                if d <= 730: return 7  # ~1y-2y bucket
                return 9
            except:
                return 9
        try:
            return order.index(p)
        except:
            return 9
    return maxp if pos(period) > pos(maxp) else period

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure simple string columns even if MultiIndex
    df.columns = [str(c).title() for c in df.columns]
    return df

# ============ Core Indicators ============

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta.clip(upper=0))
    roll_up = up.rolling(n, min_periods=1).mean()
    roll_down = down.rolling(n, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(0.0)

def swing_points(df: pd.DataFrame, lookback: int = 3):
    highs = argrelextrema(df["High"].values, np.greater_equal, order=lookback)[0]
    lows  = argrelextrema(df["Low"].values,  np.less_equal,    order=lookback)[0]
    return highs, lows

def _flatten_levels(levels) -> list[float]:
    flat = []
    for x in levels:
        if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
            arr = np.asarray(x).ravel()
            for y in arr:
                if pd.notna(y):
                    flat.append(float(y))
        else:
            if pd.notna(x):
                flat.append(float(x))
    return flat

def cluster_levels(levels, tolerance: float):
    levels = _flatten_levels(levels)
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lvl in levels[1:]:
        if abs(lvl - float(np.mean(clusters[-1]))) <= float(tolerance):
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    return [{"level": float(np.mean(c)), "touches": len(c)} for c in clusters]

def get_sr_levels(df: pd.DataFrame, lookback: int = 3, max_levels: int = 8):
    highs, lows = swing_points(df, lookback)
    res_candidates = [float(df["High"].to_numpy()[i]) for i in highs if 0 <= i < len(df)]
    sup_candidates = [float(df["Low"].to_numpy()[i])  for i in lows  if 0 <= i < len(df)]

    last_close = float(df["Close"].iloc[-1])
    last_atr = df["ATR14"].iloc[-1]
    last_atr = float(last_atr) if pd.notna(last_atr) else np.nan

    tol_candidates = [0.0075 * last_close]
    if pd.notna(last_atr) and last_atr > 0:
        tol_candidates.append(0.5 * last_atr)
    tol = float(np.max(tol_candidates))

    res = cluster_levels(res_candidates, tol)
    sup = cluster_levels(sup_candidates, tol)

    px = last_close
    res_sorted = sorted(res, key=lambda x: (-x["touches"], abs(x["level"] - px)))[:max_levels]
    sup_sorted = sorted(sup, key=lambda x: (-x["touches"], abs(x["level"] - px)))[:max_levels]
    return res_sorted, sup_sorted, tol

def breakout_signals(df: pd.DataFrame, resistances, supports, lookback_days: int = 60, vol_mult: float = 1.5):
    if df.shape[0] < max(60, lookback_days):
        return None
    recent = df.tail(lookback_days)
    last = recent.iloc[-1]
    prev = recent.iloc[-2]
    vol20 = float(np.nan_to_num(recent["VOL20"].iloc[-1], nan=0.0))

    nearest_res = min(resistances, key=lambda x: abs(x["level"] - last["Close"]))["level"] if resistances else None
    nearest_sup = min(supports,   key=lambda x: abs(x["level"] - last["Close"]))["level"] if supports else None

    bull = bear = False
    if nearest_res is not None:
        bull = (prev["Close"] <= nearest_res) and (last["Close"] > nearest_res) and (last["Volume"] > vol_mult * vol20)
    if nearest_sup is not None:
        bear = (prev["Close"] >= nearest_sup) and (last["Close"] < nearest_sup) and (last["Volume"] > vol_mult * vol20)

    return {
        "bull_breakout": bool(bull),
        "bear_breakdown": bool(bear),
        "nearest_resistance": nearest_res,
        "nearest_support": nearest_sup,
    }

# ============ Data fetch with retries & fallbacks ============

@st.cache_data(ttl=1800, show_spinner=False)
def get_data(ticker: str, period: str = "1y", interval: str = "1d", prepost: bool = False) -> pd.DataFrame:
    """
    Robust fetcher:
    1) Clamp period for intraday intervals.
    2) Try yf.download (auto_adjust=True).
    3) Fallback to Ticker().history.
    4) Retry with prepost=True if empty.
    """
    tkr = ticker.strip().upper()
    if not tkr:
        return pd.DataFrame()

    period = coerce_period_for_interval(interval, period)

    # Try primary path
    for attempt, (pp, path) in enumerate([(prepost, "download"), (True, "download"), (prepost, "history"), (True, "history")], start=1):
        try:
            if path == "download":
                df = yf.download(
                    tkr,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    prepost=pp,
                    progress=False,
                )
            else:
                df = yf.Ticker(tkr).history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    prepost=pp,
                )
        except Exception:
            df = pd.DataFrame()

        if df is not None and not df.empty:
            df = normalize_columns(df)
            # Ensure required columns
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if needed.issubset(df.columns):
                # Compute indicators
                df = df.dropna().copy()
                if df.empty:
                    continue
                df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
                df["EMA50"] = df["Close"].ewm(span=50, adjust=False, min_periods=1).mean()
                df["ATR14"] = atr(df, 14)
                df["RSI14"] = rsi(df["Close"], 14)
                df["VOL20"] = df["Volume"].rolling(20, min_periods=1).mean()
                return df

    # Still empty after all attempts
    return pd.DataFrame()

# ============ UI ============

st.title("📈 TA Scout — Support/Resistance & Breakouts")

col0, col1, col2 = st.columns([2, 1, 1])
with col0:
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
with col1:
    period = st.selectbox("Period", ["7d","14d","30d","60d","6mo","1y","2y","5y","10y","max"], index=6)
with col2:
    interval = st.selectbox("Interval", ["1d","1h","30m","15m","5m","1m"], index=0)

with st.sidebar:
    st.header("Signal Settings")
    lookback = st.slider("Swing lookback (bars)", 2, 10, 3)
    max_levels = st.slider("Max S/R levels", 4, 15, 8)
    vol_mult = st.slider("Breakout volume multiple (vs 20-bar avg)", 1.0, 3.0, 1.5, 0.1)
    show_indicators = st.checkbox("Show SMA20 / EMA50 / RSI14", value=True)
    show_atr = st.checkbox("Show ATR14 band", value=False)
    prepost = st.checkbox("Include pre/post-market (intraday)", value=False)

# Primary action
if st.button("Analyze") or ticker:
    with st.spinner("Fetching & computing…"):
        df = get_data(ticker, period=period, interval=interval, prepost=prepost)

    # If intraday combo was too long, tell user what we used
    effective_period = coerce_period_for_interval(interval, period)

    if df.empty:
        st.error(
            "No data returned after multiple attempts.\n\n"
            f"- Ticker tried: **{ticker}**\n"
            f"- Interval: **{interval}**\n"
            f"- Period requested: **{period}**, used: **{effective_period}** for intraday\n"
            "- Tips: try a different interval/period, toggle pre/post, or verify the ticker (e.g., `AAPL`, `NVDA`, `^GSPC`)."
        )
        st.stop()

    # S/R + signals
    resistances, supports, tol = get_sr_levels(df, lookback=lookback, max_levels=max_levels)
    sig = breakout_signals(df, resistances, supports, lookback_days=60, vol_mult=vol_mult)

    # ---- Chart ----
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    )])

    if show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", mode="lines"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", mode="lines"))

    if show_atr:
        upper = df["Close"] + df["ATR14"]
        lower = df["Close"] - df["ATR14"]
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="Close+ATR", mode="lines", opacity=0.4))
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="Close-ATR", mode="lines", opacity=0.4))

    for r in resistances:
        fig.add_hline(y=r["level"], line=dict(dash="dot"), annotation_text=f"R ({r['touches']})", annotation_position="right")
    for s in supports:
        fig.add_hline(y=s["level"], line=dict(dash="dot"), annotation_text=f"S ({s['touches']})", annotation_position="right")

    fig.update_layout(
        title=f"{ticker} — {effective_period} / {interval}",
        xaxis_rangeslider_visible=False,
        height=700,
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=60, r=20, b=20, l=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Tables ----
    left, right = st.columns(2)
    with left:
        st.subheader("Resistance levels")
        st.dataframe(pd.DataFrame([{"Type": "R", "Level": round(x["level"], 2), "Touches": x["touches"]} for x in resistances]), use_container_width=True)
    with right:
        st.subheader("Support levels")
        st.dataframe(pd.DataFrame([{"Type": "S", "Level": round(x["level"], 2), "Touches": x["touches"]} for x in supports]), use_container_width=True)

    # ---- Summary ----
    last = df.iloc[-1]
    st.markdown("### Signal Summary")
    vol20_last = float(np.nan_to_num(df["VOL20"].iloc[-1], nan=0.0))
    bullets = []
    bullets.append(f"**Close:** {last['Close']:.2f}   |   **Volume:** {int(last['Volume']):,}  (20-bar avg: {int(vol20_last):,})")
    bullets.append(f"**RSI14:** {last['RSI14']:.1f}   |   **ATR14:** {float(last['ATR14']):.2f}")
    if sig:
        bull = "✅ Bullish breakout" if sig["bull_breakout"] else "—"
        bear = "❌ Bearish breakdown" if sig["bear_breakdown"] else "—"
        nr = f"{sig['nearest_resistance']:.2f}" if sig['nearest_resistance'] else "n/a"
        ns = f"{sig['nearest_support']:.2f}" if sig['nearest_support'] else "n/a"
        bullets.append(f"**Nearest R:** {nr}   |   **Nearest S:** {ns}")
        bullets.append(f"**Signals:** {bull}   {bear}")
    st.write("\n\n".join(bullets))

    st.caption("Levels clustered with tolerance ≈ max(0.5×ATR, 0.75% of price). "
               f"Swing lookback={lookback}. Volume multiple={vol_mult}.")
    if is_intraday(interval) and period != effective_period:
        st.info(f"Intraday constraint: period **{period}** was clamped to **{effective_period}** for Yahoo compatibility.")

# Footer
st.caption("Educational use only — not investment advice.")
