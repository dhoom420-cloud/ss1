# ta_app.py
# ---------------------------------------------------------
# TA Scout — S/R, Breakouts, Regime, Supertrend, Squeeze, MACD, News + Cheat Sheet
# Tabs + Mobile mode + CSV + (optional) PDF (no PNG export, no Telegram)
# ---------------------------------------------------------

import io, os
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# ---------- Optional PDF (reportlab). If not installed, PDF button shows a tip ----------
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="TA Scout", layout="wide")

# Small CSS for better mobile spacing / tabs
st.markdown("""
<style>
@media (max-width: 640px) {
  .block-container {padding-top: 0.6rem; padding-bottom: 1.2rem;}
  .stTabs [role="tab"] {font-size: 0.92rem; padding: 0.25rem 0.5rem;}
  .stDataFrame, .stTable {font-size: 0.92rem;}
}
</style>
""", unsafe_allow_html=True)

# ---------- Safe secrets/env helpers ----------
def sec(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, st.secrets.get(key.upper(), default))
    except Exception:
        return os.environ.get(key) or os.environ.get(key.upper(), default)

def sec_int(key: str, default: int = 0) -> int:
    try:
        return int(sec(key, str(default)))
    except Exception:
        return default

# =================== Yahoo constraints & plotting helpers ===================

_INTRADAY_MAX_PERIOD = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d", "90m": "60d", "1h": "730d",
}

def is_intraday(interval:_
