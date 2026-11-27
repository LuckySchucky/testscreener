import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

MAX_TICKERS = 500
BATCH_SIZE = 100
INFO_WORKERS = 8

CORE_UNIVERSE = [
    "ORCL", "IREN", "ZENA", "VLN", "RUN", "EOSE", "DFLI", "BTU", "NVVE",
    "DELL", "INTC", "AAPL", "VRT", "CLS", "CRWD", "LLY", "NVDA", "MU",
    "COHR", "AMBA", "ENR.DE", "SU.PA", "005930.KS", "000660.KS"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def fetch_wiki_tables(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def extract_symbols_from_tables(url: str, suffix: str = "") -> list[str]:
    try:
        tables = fetch_wiki_tables(url)
    except Exception as e:
        print(f"Fehler beim Laden von {url}: {e}")
        return []

    best_symbols = []
    candidate_cols = ["Symbol", "Ticker", "Ticker symbol", "WKN / Symbol", "Epic"]

    for df in tables:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join([str(c) for c in col]).strip() for col in df.columns]

        matched_col = None
        for col in df.columns:
            col_str = str(col).strip()
            if any(c.lower() == col_str.lower() or c.lower() in col_str.lower() for c in candidate_cols):
                matched_col = col
                break

        if matched_col is not None and len(df) >= 15:
            raw_vals = df[matched_col].dropna().astype(str).tolist()
            cleaned = []
            for sym in raw_vals:
                sym = sym.split("[")[0].strip().upper()
                if not sym or sym in ("NAN", "NONE", "-"):
                    continue
                if suffix == ".DE":
                    sym = sym.replace(".DE", "") + ".DE"
                elif suffix == ".PA":
                    sym = sym.replace(".PA", "") + ".PA"
                else:
                    if "." in sym and not sym.endswith((".DE", ".PA", ".KS", ".L", ".SW")):
                        sym = sym.replace(".", "-")
                cleaned.append(sym)

            if len(cleaned) > len(best_symbols):
                best_symbols = cleaned

    print(f"{url} -> {len(best_symbols)} Ticker extrahiert")
    return best_symbols


def build_ticker_universe() -> list[str]:
    sp500 = extract_symbols_from_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    nasdaq100 = extract_symbols_from_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
    dax = extract_symbols_from_tables("https://en.wikipedia.org/wiki/DAX", suffix=".DE")
    mdax = extract_symbols_from_tables("https://en.wikipedia.org/wiki/MDAX", suffix=".DE")
    cac40 = extract_symbols_from_tables("https://en.wikipedia.org/wiki/CAC_40", suffix=".PA")

    combined = []
    seen = set()

    for sym in CORE_UNIVERSE + sp500 + nasdaq100 + dax + mdax + cac40:
        if sym and sym not in seen:
            seen.add(sym)
            combined.append(sym)

    print(f"Gesamtpool vor Limit: {len(combined)} eindeutige Ticker")
    return combined[:MAX_TICKERS]


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 1) if pd.notna(val) else 50.0


def _extract_single_history(df: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    if not isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns:
            return None
        sub = pd.DataFrame({
            "Close": df["Close"],
            "Volume": df["Volume"] if "Volume" in df.columns else 0.0,
        }).dropna(subset=["Close"])
        return sub if len(sub) >= 60 else None

    lvl0 = set(df.columns.get_level_values(0))
    lvl1 = set(df.columns.get_level_values(1))

    if sym in lvl0:
        block = df[sym]
    elif "Close" in lvl0:
        block = pd.DataFrame({
            "Close": df["Close"][sym] if sym in df["Close"].columns else np.nan,
            "Volume": df["Volume"][sym] if ("Volume" in lvl0 and sym in df["Volume"].columns) else 0.0,
        })
    elif "Close" in lvl1:
        cols = {}
        for field in ("Close", "Volume"):
            matches = [c for c in df.columns if c[1] == field and (c[0] == sym or len(lvl0) == 1)]
            if matches:
                cols[field] = df[matches[0]]
        if "Close" not in cols:
            return None
        block = pd.DataFrame(cols)
    else:
        return None

    if "Close" not in block:
        return None

    close_s = block["Close"]
    if isinstance(close_s, pd.DataFrame):
        close_s = close_s.iloc[:, 0]

    if "Volume" in block:
        vol_s = block["Volume"]
        if isinstance(vol_s, pd.DataFrame):
            vol_s = vol_s.iloc[:, 0]
    else:
        vol_s = pd.Series(0.0, index=close_s.index)

    sub = pd.DataFrame({"Close": close_s, "Volume": vol_s}).dropna(subset=["Close"])
    return sub if len(sub) >= 60 else None


def download_history_batches(tickers: list[str]) -> dict[str, pd.DataFrame]:
    all_tickers = list(dict.fromkeys(["^GSPC"] + tickers))
    history_map: dict[str, pd.DataFrame] = {}

    for i in range(0, len(all_tickers), BATCH_SIZE):
        chunk = all_tickers[i:i + BATCH_SIZE]
        print(f"Lade Kurs-Batch {i // BATCH_SIZE + 1}/{(len(all_tickers) - 1) // BATCH_SIZE + 1} ({len(chunk)} Ticker)...")
        try:
            df = yf.download(
                chunk,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            for sym in chunk:
                sub = _extract_single_history(df, sym)
                if sub is not None:
                    history_map[sym] = sub
        except Exception as e:
            print(f"Batch-Fehler bei Block {i}: {e}")
        time.sleep(0.5)

    return history_map


def fetch_info_single(sym: str) -> tuple[str, dict]:
    time.sleep(random.uniform(0.05, 0.2))
    try:
        info = yf.Ticker(sym).info or {}
        return sym, info
    except Exception:
        return sym, {}


def score_stock(sym: str, hist: pd.DataFrame, info: dict, sp500_3m_ret: float, core_set: set[str]) -> dict | None:
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)

    if len(close) < 65:
