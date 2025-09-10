# ta_app.py
# ---------------------------------------------------------
# TA Scout — S/R, Breakouts, Regime, Supertrend, Squeeze, MACD, News + Cheat Sheet (Tabs)
# ---------------------------------------------------------
# pip install streamlit yfinance pandas numpy plotly
# (optional for RSS fallback in News): pip install feedparser

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TA Scout", layout="wide")

# =================== Yahoo constraints & plotting helpers ===================

_INTRADAY_MAX_PERIOD = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",  # 1h
    "90m": "60d",
    "1h": "730d",
}

def is_intraday(interval: str) -> bool:
    return interval.endswith(("m", "h"))

def interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    return 1440  # 1d+

def coerce_period_for_interval(interval: str, period: str) -> str:
    """Clamp period for intraday intervals to avoid empty frames."""
    if not is_intraday(interval):
        return period
    maxp = _INTRADAY_MAX_PERIOD.get(interval, "60d")
    order = ["7d","14d","30d","45d","60d","90d","180d","1y","730d","2y","5y","10y","max"]
    def pos(p):
        p = p.lower()
        if p.endswith("d"):
            try:
                d = int(p[:-1])
                if d <= 7: return 0
                if d <= 14: return 1
                if d <= 30: return 2
                if d <= 45: return 3
                if d <= 60: return 4
                if d <= 90: return 5
                if d <= 180: return 6
                if d <= 730: return 8
                return 10
            except:
                return 10
        try:
            return order.index(p)
        except:
            return 10
    return maxp if pos(period) > pos(maxp) else period

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]  # handle MultiIndex→strings
    return df

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    # Map alternates to OHLCV if needed
    if "Close" not in df.columns:
        for alt in ["Adj Close", "Adjusted Close", "Close*"]:
            if alt in df.columns:
                df["Close"] = df[alt]; break
    if "Open" not in df.columns:
        for alt in ["Opening Price", "Open*"]:
            if alt in df.columns:
                df["Open"] = df[alt]; break
    if "High" not in df.columns:
        for alt in ["High Price", "High*"]:
            if alt in df.columns:
                df["High"] = df[alt]; break
    if "Low" not in df.columns:
        for alt in ["Low Price", "Low*"]:
            if alt in df.columns:
                df["Low"] = df[alt]; break
    if "Volume" not in df.columns:
        df["Volume"] = np.nan  # indices often miss volume

    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    core = {"Open","High","Low","Close"}
    if not core.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=list(core))
    df["Volume"] = df["Volume"].fillna(0.0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def plotly_x(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    try:
        if getattr(idx, "tz", None) is not None:
            return idx.tz_convert("UTC").tz_localize(None)
    except Exception:
        pass
    return pd.DatetimeIndex(idx)

# =================== Core indicators ===================

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0); dn = (-d.clip(upper=0))
    roll_up = up.rolling(n, min_periods=1).mean()
    roll_dn = dn.rolling(n, min_periods=1).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    return (100 - (100/(1+rs))).fillna(0.0)

# =================== S/R (swing clustering) ===================

def swing_points(df: pd.DataFrame, lookback: int = 3):
    win = 2*lookback + 1
    h, l = df["High"], df["Low"]
    h_max = h.rolling(win, center=True, min_periods=1).max()
    l_min = l.rolling(win, center=True, min_periods=1).min()
    high_idx = np.where(h.values == h_max.values)[0]
    low_idx  = np.where(l.values == l_min.values)[0]
    return high_idx, low_idx

def _flatten_levels(levels):
    flat = []
    for x in levels:
        if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
            arr = np.asarray(x).ravel()
            for y in arr:
                if pd.notna(y): flat.append(float(y))
        else:
            if pd.notna(x): flat.append(float(x))
    return flat

def cluster_levels(levels, tolerance: float):
    levels = _flatten_levels(levels)
    if not levels: return []
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
    res_cands = [float(df["High"].to_numpy()[i]) for i in highs if 0 <= i < len(df)]
    sup_cands = [float(df["Low"].to_numpy()[i])  for i in lows  if 0 <= i < len(df)]
    last_close = float(df["Close"].iloc[-1])
    last_atr = float(df["ATR14"].iloc[-1]) if pd.notna(df["ATR14"].iloc[-1]) else np.nan
    tol = float(np.max([0.0075*last_close] + ([0.5*last_atr] if pd.notna(last_atr) and last_atr>0 else [])))
    res = cluster_levels(res_cands, tol)
    sup = cluster_levels(sup_cands, tol)
    px = last_close
    res_sorted = sorted(res, key=lambda x: (-x["touches"], abs(x["level"]-px)))[:max_levels]
    sup_sorted = sorted(sup, key=lambda x: (-x["touches"], abs(x["level"]-px)))[:max_levels]
    return res_sorted, sup_sorted, tol

# =================== Breakouts ===================

def bollinger_bands(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n, min_periods=1).mean()
    sd = close.rolling(n, min_periods=1).std()
    return ma, ma + k*sd, ma - k*sd

def cross_up(price: pd.Series, level: pd.Series) -> pd.Series:
    return (price > level) & (price.shift(1) <= level.shift(1))

def cross_dn(price: pd.Series, level: pd.Series) -> pd.Series:
    return (price < level) & (price.shift(1) >= level.shift(1))

def donchian_breakout(df: pd.DataFrame, n: int, vol_mult: float):
    hi_prev = df["High"].rolling(n, min_periods=1).max().shift(1)
    lo_prev = df["Low"].rolling(n,  min_periods=1).min().shift(1)
    vol20 = df["VOL20"].fillna(0.0)
    up = cross_up(df["Close"], hi_prev) & (df["Volume"] > vol_mult*vol20)
    dn = cross_dn(df["Close"], lo_prev) & (df["Volume"] > vol_mult*vol20)
    return {"up": up, "dn": dn, "lvl_up": hi_prev, "lvl_dn": lo_prev}

def fiftytwo_week_breakout(df: pd.DataFrame, vol_mult: float, bars: int = 252):
    hi_prev = df["High"].rolling(bars, min_periods=1).max().shift(1)
    lo_prev = df["Low"].rolling(bars,  min_periods=1).min().shift(1)
    vol20 = df["VOL20"].fillna(0.0)
    up = cross_up(df["Close"], hi_prev) & (df["Volume"] > vol_mult*vol20)
    dn = cross_dn(df["Close"], lo_prev) & (df["Volume"] > vol_mult*vol20)
    return {"up": up, "dn": dn, "lvl_up": hi_prev, "lvl_dn": lo_prev}

def bollinger_breakout(df: pd.DataFrame, n: int, k: float, vol_mult: float):
    _, upper, lower = bollinger_bands(df["Close"], n=n, k=k)
    vol20 = df["VOL20"].fillna(0.0)
    up = cross_up(df["Close"], upper) & (df["Volume"] > vol_mult*vol20)
    dn = cross_dn(df["Close"], lower) & (df["Volume"] > vol_mult*vol20)
    return {"up": up, "dn": dn, "lvl_up": upper, "lvl_dn": lower}

def sr_breakout_lastbar(df, resistances, supports, vol_mult: float = 1.5):
    if df.shape[0] < 2: return None
    last, prev = df.iloc[-1], df.iloc[-2]
    vol20 = float(np.nan_to_num(df["VOL20"].iloc[-1], nan=0.0))
    nearest_res = min(resistances, key=lambda x: abs(x["level"]-last["Close"]))["level"] if resistances else None
    nearest_sup = min(supports,   key=lambda x: abs(x["level"]-last["Close"]))["level"] if supports else None
    bull = bear = False
    if nearest_res is not None:
        bull = (prev["Close"] <= nearest_res) and (last["Close"] > nearest_res) and (last["Volume"] > vol20*vol_mult)
    if nearest_sup is not None:
        bear = (prev["Close"] >= nearest_sup) and (last["Close"] < nearest_sup) and (last["Volume"] > vol20*vol_mult)
    return {"type":"S/R","bull":bool(bull),"bear":bool(bear),"level_up":nearest_res,"level_dn":nearest_sup}

# =================== Regime, Supertrend, Squeeze, MACD ===================

def regime_series(df, fast=50, slow=200):
    ema_fast = df["Close"].ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False, min_periods=1).mean()
    return ema_fast, ema_slow, (ema_fast > ema_slow)

def supertrend(df, n=10, m=3.0):
    hl2 = (df["High"] + df["Low"]) / 2.0
    atrn = atr(df, n)
    upperband = hl2 + m * atrn
    lowerband = hl2 - m * atrn

    fub = upperband.copy()
    flb = lowerband.copy()
    for i in range(1, len(df)):
        fub.iat[i] = min(upperband.iat[i], fub.iat[i-1]) if df["Close"].iat[i-1] > fub.iat[i-1] else upperband.iat[i]
        flb.iat[i] = max(lowerband.iat[i], flb.iat[i-1]) if df["Close"].iat[i-1] < flb.iat[i-1] else lowerband.iat[i]

    st_line = pd.Series(index=df.index, dtype="float64")
    dir_up = True
    for i in range(len(df)):
        if dir_up:
            st_line.iat[i] = flb.iat[i]
            if df["Close"].iat[i] < flb.iat[i]:
                dir_up = False
        else:
            st_line.iat[i] = fub.iat[i]
            if df["Close"].iat[i] > fub.iat[i]:
                dir_up = True
    trend_up = df["Close"] > st_line
    return st_line, trend_up

def keltner_channels(df, n=20, m=1.5):
    ema = df["Close"].ewm(span=n, adjust=False, min_periods=1).mean()
    kc_u = ema + m * atr(df, n)
    kc_l = ema - m * atr(df, n)
    return ema, kc_u, kc_l

def squeeze_on(df, bb_n=20, bb_k=2.0, kc_n=20, kc_m=1.5):
    ma, bb_u, bb_l = bollinger_bands(df["Close"], n=bb_n, k=bb_k)
    _, kc_u, kc_l = keltner_channels(df, n=kc_n, m=kc_m)
    on  = (bb_u < kc_u) & (bb_l > kc_l)
    off = (bb_u > kc_u) & (bb_l < kc_l)
    return on.fillna(False), off.fillna(False), (ma, bb_u, bb_l, kc_u, kc_l)

def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_s = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    line = ema_f - ema_s
    sig  = line.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = line - sig
    return line, sig, hist

# =================== News helpers ===================

def _thumb_from_yf_item(it):
    try:
        res = (it.get("thumbnail") or {}).get("resolutions") or []
        if res:
            res = sorted(res, key=lambda r: r.get("width", 0), reverse=True)
            return res[0].get("url")
    except Exception:
        pass
    return None

def _to_et_timestamp(ts):
    try:
        return pd.Timestamp.utcfromtimestamp(int(ts)).tz_localize("UTC").tz_convert("America/New_York")
    except Exception:
        return pd.NaT

def get_news_yf(ticker: str, limit: int = 8):
    rows, seen = [], set()
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []
    for it in items:
        title = it.get("title") or ""
        link = it.get("link") or it.get("url") or ""
        if not title or not link:
            continue
        key = (title, link)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "title": title,
            "link": link,
            "publisher": it.get("publisher") or "",
            "published": _to_et_timestamp(it.get("providerPublishTime")),
            "thumbnail": _thumb_from_yf_item(it),
        })
        if len(rows) >= limit:
            break
    return rows

def get_news_rss(ticker: str, limit: int = 8):
    try:
        import feedparser, urllib.parse
    except Exception:
        return []
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(ticker)}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    rows = []
    for e in (feed.entries or [])[: limit * 2]:
        title = getattr(e, "title", "")
        link = getattr(e, "link", "")
        if not title or not link:
            continue
        dt = None
        try:
            if getattr(e, "published_parsed", None):
                dt = pd.Timestamp(*e.published_parsed[:6], tz="UTC").tz_convert("America/New_York")
        except Exception:
            dt = None
        rows.append({
            "title": title,
            "link": link,
            "publisher": getattr(getattr(e, "source", None) or {}, "title", None) or "Yahoo Finance",
            "published": dt if dt is not None else pd.NaT,
            "thumbnail": None,
        })
        if len(rows) >= limit:
            break
    return rows

def get_news(ticker: str, limit: int = 8):
    rows = get_news_yf(ticker, limit=limit)
    if not rows:
        rows = get_news_rss(ticker, limit=limit)
    try:
        rows.sort(key=lambda r: r.get("published") or pd.Timestamp.min, reverse=True)
    except Exception:
        pass
    return rows

def render_news(rows):
    if not rows:
        st.info("No recent news found for this ticker.")
        return
    for r in rows:
        c1, c2 = st.columns([1, 5])
        with c1:
            if r.get("thumbnail"):
                st.image(r["thumbnail"], use_container_width=True)
        with c2:
            st.markdown(f"**[{r['title']}]({r['link']})**")
            meta = []
            if r.get("publisher"): meta.append(r["publisher"])
            ts = r.get("published")
            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                meta.append(ts.strftime("%b %d, %Y %I:%M %p %Z"))
            if meta: st.caption(" · ".join(meta))
        st.divider()

# =================== Cheat Sheet ===================

def render_cheatsheet():
    with st.expander("📘 Technical Indicator Cheat Sheet (tap to open)", expanded=False):
        st.markdown("""
### Core Prices & Volume
- **OHLCV**: Open, High, Low, Close, Volume per bar. Many signals also reference **VOL20** (20-bar average volume).

---

### Trend & Averages
- **SMA(n)** — Simple Moving Average  
  Average of last _n_ closes.  
  `SMA_n = (Σ Close) / n`

- **EMA(n)** — Exponential Moving Average  
  Weights recent prices more.  
  `EMA_t = α·Close_t + (1−α)·EMA_{t−1}`, `α = 2/(n+1)`  
  **Use**: Trend filter (e.g., **Regime** = EMA50 > EMA200).

---

### Volatility
- **ATR(n)** — Average True Range  
  `TR = max(High−Low, |High−PrevClose|, |Low−PrevClose|)`; `ATR_n` is rolling avg of TR.  
  **Use**: Position sizing, stops, S/R clustering tolerance.

- **Bollinger Bands (BB)**  
  `MA = SMA(Close, n)`; **Upper/Lower** = `MA ± k·σ`.  
  **Breakout** when Close crosses bands (with volume).

- **Keltner Channels (KC)**  
  `Mid = EMA(Close, n)`; **Upper/Lower** = `Mid ± m·ATR(n)`.  
  **Squeeze** when BB are inside KC ⇒ contraction.

---

### Momentum
- **RSI(n)** — Relative Strength Index  
  `RS = AvgGain / AvgLoss`; `RSI = 100 − 100/(1+RS)`.

- **MACD (12,26,9)**  
  `MACD = EMA_12 − EMA_26`; **Signal** = EMA_9(MACD); **Hist** = MACD − Signal.  
  **Read**: Histogram flip ↑ often confirms bullish momentum.

---

### Breakouts & Levels
- **Support/Resistance (S/R)**  
  From swing highs/lows, clustered within tolerance ≈ `max(0.5·ATR, 0.75% of price)`.  
  **Breakout/Down** requires **volume spike** vs VOL20.

- **Donchian (n)**  
  Highest high / lowest low over last _n_ bars (shifted).  
  Cross with volume ⇒ breakout.

- **52-Week Breakout**  
  Donchian with `n≈252` trading days.

- **ORB (intraday)**  
  First X minutes range; break with volume confirms move.

---

### Trend-Following Stops
- **Supertrend (n, m)**  
  Uses `HL2 ± m·ATR(n)`; flips when price pierces the active band.

- **Chandelier Exit (n, k)** (optional)  
  Long stop = `HighestHigh_n − k·ATR(n)`.

---

### Volume & Confirmation
- **VOL20** = 20-bar avg volume; we gate breakouts with `Volume > vol_mult × VOL20`.

---

### Playbooks
- **Breakout + Regime**: Donchian(20) ↑ **and** EMA50>EMA200 **and** Volume > 1.5×VOL20.  
- **Squeeze Release**: Squeeze **ON** → **OFF** + MACD hist flip ↑ + close above KC mid / Supertrend.  
- **Intraday**: ORB High break above VWAP (if added) with strong volume.

*Educational use only — not investment advice.*
        """)

# =================== Data fetch with retries & fallbacks ===================

@st.cache_data(ttl=1800, show_spinner=False)
def get_data(ticker: str, period: str = "1y", interval: str = "1d", prepost: bool = False) -> pd.DataFrame:
    tkr = ticker.strip().upper()
    if not tkr: return pd.DataFrame()
    period = coerce_period_for_interval(interval, period)

    for (pp, path) in [(prepost, "download"), (True, "download"), (prepost, "history"), (True, "history")]:
        try:
            if path == "download":
                df = yf.download(tkr, period=period, interval=interval, auto_adjust=True, prepost=pp, progress=False)
            else:
                df = yf.Ticker(tkr).history(period=period, interval=interval, auto_adjust=True, prepost=pp)
        except Exception:
            df = pd.DataFrame()

        if df is not None and not df.empty:
            df = clean_ohlcv(df)
            if df.empty: continue
            # Core indicators (+ EMA200 for regime)
            df["SMA20"]  = df["Close"].rolling(20, min_periods=1).mean()
            df["EMA50"]  = df["Close"].ewm(span=50,  adjust=False, min_periods=1).mean()
            df["EMA200"] = df["Close"].ewm(span=200, adjust=False, min_periods=1).mean()
            df["ATR14"]  = atr(df, 14)
            df["RSI14"]  = rsi(df["Close"], 14)
            df["VOL20"]  = df["Volume"].rolling(20, min_periods=1).mean()
            return df
    return pd.DataFrame()

# =================== UI ===================

st.title("📈 TA Scout — S/R, Breakouts, Regime, Supertrend, Squeeze, MACD, News + Cheat Sheet")

col0, col1, col2 = st.columns([2, 1, 1])
with col0:
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
with col1:
    period = st.selectbox("Period", ["7d","14d","30d","60d","6mo","1y","2y","5y","10y","max"], index=5)
with col2:
    interval = st.selectbox("Interval", ["1d","1h","30m","15m","5m","1m"], index=0)

with st.sidebar:
    st.header("Signal Settings")
    lookback = st.slider("Swing lookback (bars)", 2, 10, 3)
    max_levels = st.slider("Max S/R levels", 4, 15, 8)
    vol_mult = st.slider("Breakout volume multiple (vs 20-bar avg)", 1.0, 3.0, 1.5, 0.1)
    prepost = st.checkbox("Include pre/post-market (intraday)", value=False)

    st.markdown("---")
    st.subheader("Breakout Indicators")
    dc20_on = st.checkbox("Donchian (fast)", value=True)
    dc20_n  = st.number_input("Donchian N (fast)", value=20, min_value=5, max_value=200, step=1)
    dc55_on = st.checkbox("Donchian 55 (slow)", value=True)
    dc55_n  = st.number_input("Donchian N (slow)", value=55, min_value=10, max_value=400, step=1)
    w52_on  = st.checkbox("52-week breakout (252 bars)", value=True)
    bb_on   = st.checkbox("Bollinger breakout", value=True)
    bb_n    = st.number_input("BB window (n)", value=20, min_value=10, max_value=200, step=1)
    bb_k    = st.number_input("BB std (k)", value=2.0, min_value=1.0, max_value=3.5, step=0.1)

    st.markdown("---")
    st.subheader("Regime / Trend & Momentum")
    regime_on = st.checkbox("Apply Regime Filter (EMA50>EMA200)", value=True)
    regime_fast = st.number_input("Regime EMA fast", value=50, min_value=5, max_value=200, step=5)
    regime_slow = st.number_input("Regime EMA slow", value=200, min_value=20, max_value=400, step=10)
    st_on = st.checkbox("Show Supertrend", value=True)
    st_n  = st.number_input("Supertrend ATR period (n)", value=10, min_value=5, max_value=50, step=1)
    st_m  = st.number_input("Supertrend multiplier (m)", value=3.0, min_value=1.0, max_value=5.0, step=0.1)
    sq_on = st.checkbox("Mark Squeeze (BB in KC)", value=True)
    sq_bb_n = st.number_input("Squeeze BB window", value=20, min_value=10, max_value=200, step=1)
    sq_bb_k = st.number_input("Squeeze BB std k", value=2.0, min_value=1.0, max_value=3.5, step=0.1)
    sq_kc_n = st.number_input("Keltner EMA window", value=20, min_value=10, max_value=200, step=1)
    sq_kc_m = st.number_input("Keltner ATR mult", value=1.5, min_value=1.0, max_value=3.0, step=0.1)
    macd_on = st.checkbox("Show MACD panel", value=True)
    macd_fast = st.number_input("MACD fast", value=12, min_value=2, max_value=50, step=1)
    macd_slow = st.number_input("MACD slow", value=26, min_value=5, max_value=100, step=1)
    macd_sig  = st.number_input("MACD signal", value=9, min_value=2, max_value=50, step=1)

tabs = st.tabs(["📊 Chart", "🧱 Levels & Signals", "📰 News", "📘 Cheat Sheet"])

# Primary action
if st.button("Analyze") or ticker:
    with st.spinner("Fetching & computing…"):
        df = get_data(ticker, period=period, interval=interval, prepost=prepost)

    effective_period = coerce_period_for_interval(interval, period)

    if df.empty:
        st.error(
            "No data returned.\n\n"
            f"- Ticker: **{ticker}** | Interval: **{interval}** | Requested Period: **{period}** → Used: **{effective_period}** (intraday clamp)\n"
            "- Try a different interval/period, toggle pre/post, or verify the ticker (e.g., AAPL, NVDA, ^GSPC)."
        )
        st.stop()

    # ------- Compute S/R -------
    resistances, supports, tol = get_sr_levels(df, lookback=lookback, max_levels=max_levels)
    sr_sig = sr_breakout_lastbar(df, resistances, supports, vol_mult=vol_mult)

    # ------- Breakouts -------
    breakout_series = {}
    if dc20_on: breakout_series["DC20"] = donchian_breakout(df, n=int(dc20_n), vol_mult=vol_mult)
    if dc55_on: breakout_series["DC55"] = donchian_breakout(df, n=int(dc55_n), vol_mult=vol_mult)
    if bb_on:   breakout_series["BB"]   = bollinger_breakout(df, n=int(bb_n), k=float(bb_k), vol_mult=vol_mult)
    if w52_on:  breakout_series["W52"]  = fiftytwo_week_breakout(df, vol_mult=vol_mult, bars=252)

    # ------- Regime / Supertrend / Squeeze / MACD -------
    ema_fast, ema_slow, regime = regime_series(df, fast=int(regime_fast), slow=int(regime_slow))
    regime_now = bool(regime.iloc[-1])

    st_line, st_up = supertrend(df, n=int(st_n), m=float(st_m))
    st_cross = st_up != st_up.shift(1)

    sq_on_mask, sq_off_mask, (bb_ma, bb_u, bb_l, kc_u, kc_l) = squeeze_on(
        df, bb_n=int(sq_bb_n), bb_k=float(sq_bb_k), kc_n=int(sq_kc_n), kc_m=float(sq_kc_m)
    )

    macd_line, macd_sig_line, macd_hist = macd(df["Close"], fast=int(macd_fast), slow=int(macd_slow), signal=int(macd_sig))
    macd_bull_flip = (macd_hist.iloc[-1] > 0) and (macd_hist.iloc[-2] <= 0) if len(macd_hist) >= 2 else False
    macd_bear_flip = (macd_hist.iloc[-1] < 0) and (macd_hist.iloc[-2] >= 0) if len(macd_hist) >= 2 else False

    # -------------------- TAB 1: CHART --------------------
    with tabs[0]:
        xidx = plotly_x(df.index)
        fig = go.Figure(data=[go.Candlestick(
            x=xidx, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
        )])

        # EMAs for regime
        fig.add_trace(go.Scatter(x=xidx, y=ema_fast, name=f"EMA{int(regime_fast)}", mode="lines"))
        fig.add_trace(go.Scatter(x=xidx, y=ema_slow, name=f"EMA{int(regime_slow)}", mode="lines"))

        # Supertrend line + flips
        if st_on:
            fig.add_trace(go.Scatter(x=xidx, y=st_line, name="Supertrend", mode="lines"))
            recent = min(len(df), 400)
            flip_mask = st_cross.iloc[-recent:].fillna(False).to_numpy()
            if flip_mask.any():
                y_slice = df["Close"].iloc[-recent:]
                x_slice = xidx[-recent:]
                fig.add_trace(go.Scatter(
                    x=x_slice[flip_mask], y=y_slice[flip_mask],
                    mode="markers", name="ST Flip",
                    marker_symbol="x", marker_size=10,
                    hovertemplate="Supertrend flip<extra></extra>"
                ))

        # Optional Squeeze guides + dots
        if sq_on:
            fig.add_trace(go.Scatter(x=xidx, y=bb_u, name="BB Upper", mode="lines", opacity=0.2, showlegend=False))
            fig.add_trace(go.Scatter(x=xidx, y=bb_l, name="BB Lower", mode="lines", opacity=0.2, showlegend=False))
            fig.add_trace(go.Scatter(x=xidx, y=kc_u, name="KC Upper", mode="lines", opacity=0.15, showlegend=False))
            fig.add_trace(go.Scatter(x=xidx, y=kc_l, name="KC Lower", mode="lines", opacity=0.15, showlegend=False))
            recent = min(len(df), 400)
            m = sq_on_mask.iloc[-recent:].to_numpy()
            if m.any():
                fig.add_trace(go.Scatter(
                    x=xidx[-recent:][m],
                    y=df["Low"].iloc[-recent:][m]*0.995,
                    mode="markers", name="Squeeze ON",
                    marker_symbol="circle", marker_size=8,
                    hovertemplate="Squeeze ON<extra></extra>"
                ))

        # Plot S/R hlines
        for r in resistances:
            fig.add_hline(y=r["level"], line=dict(dash="dot"),
                          annotation_text=f"R ({r['touches']})", annotation_position="right")
        for s in supports:
            fig.add_hline(y=s["level"], line=dict(dash="dot"),
                          annotation_text=f"S ({s['touches']})", annotation_position="right")

        # Breakout markers (recent window)
        NMARK = min(len(df), 400)
        idx_slice = xidx[-NMARK:]
        y_slice = df["Close"].iloc[-NMARK:]

        def add_markers(name, res, up_label, dn_label):
            if not res: return
            up_mask = res["up"].iloc[-NMARK:].fillna(False).to_numpy()
            dn_mask = res["dn"].iloc[-NMARK:].fillna(False).to_numpy()
            if up_mask.any():
                fig.add_trace(go.Scatter(
                    x=idx_slice[up_mask], y=y_slice[up_mask],
                    mode="markers+text", name=f"{name} ↑",
                    marker_symbol="triangle-up", marker_size=12,
                    text=[up_label]*int(up_mask.sum()), textposition="top center",
                    hovertemplate=f"{name} breakout↑<extra></extra>", legendgroup="breakouts"
                ))
            if dn_mask.any():
                fig.add_trace(go.Scatter(
                    x=idx_slice[dn_mask], y=y_slice[dn_mask],
                    mode="markers+text", name=f"{name} ↓",
                    marker_symbol="triangle-down", marker_size=12,
                    text=[dn_label]*int(dn_mask.sum()), textposition="bottom center",
                    hovertemplate=f"{name} breakdown↓<extra></extra>", legendgroup="breakouts"
                ))

        if "DC20" in breakout_series: add_markers("DC20", breakout_series["DC20"], "DC20↑", "DC20↓")
        if "DC55" in breakout_series: add_markers("DC55", breakout_series["DC55"], "DC55↑", "DC55↓")
        if "BB"   in breakout_series: add_markers("BB",   breakout_series["BB"],   "BB↑",   "BB↓")
        if "W52"  in breakout_series: add_markers("W52",  breakout_series["W52"],  "W52↑",  "W52↓")

        fig.update_layout(
            title=f"{ticker} — {effective_period} / {interval}",
            xaxis_rangeslider_visible=False,
            height=740,
            legend=dict(orientation="h", y=1.02),
            margin=dict(t=60, r=20, b=20, l=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # MACD panel
        if macd_on:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=xidx, y=macd_hist, name="MACD Hist", opacity=0.4))
            fig_macd.add_trace(go.Scatter(x=xidx, y=macd_line, name="MACD", mode="lines"))
            fig_macd.add_trace(go.Scatter(x=xidx, y=macd_sig_line, name="Signal", mode="lines"))
            fig_macd.update_layout(height=240, margin=dict(t=10, r=20, b=10, l=20), showlegend=True)
            st.plotly_chart(fig_macd, use_container_width=True)

    # -------------------- TAB 2: LEVELS & SIGNALS --------------------
    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.subheader("Resistance levels")
            st.dataframe(pd.DataFrame([{"Type":"R","Level":round(x["level"],2),"Touches":x["touches"]} for x in resistances]),
                         use_container_width=True)
        with right:
            st.subheader("Support levels")
            st.dataframe(pd.DataFrame([{"Type":"S","Level":round(x["level"],2),"Touches":x["touches"]} for x in supports]),
                         use_container_width=True)

        # Signal Summary (last bar)
        st.markdown("### Signal Summary (last bar)")
        last = df.iloc[-1]
        vol20_last = float(np.nan_to_num(df["VOL20"].iloc[-1], nan=0.0))

        lines = []
        lines.append(f"**Close:** {last['Close']:.2f}   |   **Volume:** {int(last['Volume']):,}  (20-bar avg: {int(vol20_last):,})")
        lines.append(f"**RSI14:** {last['RSI14']:.1f}   |   **ATR14:** {float(last['ATR14']):.2f}")

        # Regime
        regime_txt = "✅ Bullish (EMA fast above slow)" if regime_now else "❌ Bearish (EMA fast below slow)"
        lines.append(f"**Regime:** {regime_txt}")

        # Supertrend
        st_txt = "✅ Supertrend UP" if bool(st_up.iloc[-1]) else "❌ Supertrend DOWN"
        lines.append(f"**Supertrend:** {st_txt}")

        # Squeeze
        sq_txt = "⏳ Squeeze ON" if bool(sq_on_mask.iloc[-1]) else ("✅ Squeeze OFF (released)" if bool(sq_off_mask.iloc[-1]) else "—")
        lines.append(f"**Squeeze:** {sq_txt}")

        # MACD flips
        if macd_bull_flip:
            lines.append("**MACD:** ✅ Histogram turned **positive**")
        elif macd_bear_flip:
            lines.append("**MACD:** ❌ Histogram turned **negative**")
        else:
            lines.append(f"**MACD:** Hist {('>0' if macd_hist.iloc[-1] > 0 else '<0')} (no fresh flip)")

        # S/R signal (optionally filtered by regime)
        if sr_sig:
            nr = f"{sr_sig['level_up']:.2f}" if sr_sig['level_up'] else "n/a"
            ns = f"{sr_sig['level_dn']:.2f}" if sr_sig['level_dn'] else "n/a"
            srbull = sr_sig["bull"] and (regime_now if regime_on else True)
            srbear = sr_sig["bear"] and ((not regime_now) if regime_on else True)
            lines.append(f"**Nearest R/S:** {nr} / {ns}   |   **S/R Signals (regime-applied={regime_on}):** {'✅ Breakout' if srbull else '—'}   {'❌ Breakdown' if srbear else '—'}")

        # Breakouts (last bar), with regime filter if on
        for name, res in breakout_series.items():
            lb_up = bool(res["up"].iloc[-1]) if len(res["up"]) else False
            lb_dn = bool(res["dn"].iloc[-1]) if len(res["dn"]) else False
            if regime_on:
                if lb_up and not regime_now: lb_up = False
                if lb_dn and regime_now: lb_dn = False
            lu = res["lvl_up"].iloc[-1] if pd.notna(res["lvl_up"].iloc[-1]) else None
            ld = res["lvl_dn"].iloc[-1] if pd.notna(res["lvl_dn"].iloc[-1]) else None
            lines.append(f"**{name}:** up {'✅' if lb_up else '—'} @ {f'{lu:.2f}' if lu else 'n/a'}   |   down {'❌' if lb_dn else '—'} @ {f'{ld:.2f}' if ld else 'n/a'}")

        st.write("\n\n".join(lines))
        st.caption("Regime filter gates signals to trend direction. Squeeze marks volatility contraction. Educational use only.")

    # -------------------- TAB 3: NEWS --------------------
    with tabs[2]:
        with st.spinner("Loading news…"):
            news_rows = get_news(ticker, limit=8)
        render_news(news_rows)

    # -------------------- TAB 4: CHEAT SHEET --------------------
    with tabs[3]:
        render_cheatsheet()

# Footer
st.caption("© TA Scout — for education, not investment advice.")
