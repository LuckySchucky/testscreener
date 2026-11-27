import json
import math
import time
import random
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

# Deine priorisierte CORE_UNIVERSE (immer zuerst drin)
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
        # MultiIndex-Spalten flachklopfen falls vorhanden
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
                    # US-Ticker: BRK.B -> BRK-B für yfinance
                    if "." in sym and not sym.endswith((".DE", ".PA", ".KS", ".L", ".SW")):
                        sym = sym.replace(".", "-")
                cleaned.append(sym)

            # Größte gefundene Komponententabelle merken (nicht frühzeitig abbrechen!)
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

    # Core-Werte zuerst, danach breite Indizes mischen
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


def download_history_batches(tickers: list[str]) -> dict[str, pd.DataFrame]:
    all_tickers = list(dict.fromkeys(["^GSPC"] + tickers))
    history_map = {}

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
            if df.empty:
                continue

            if len(chunk) == 1:
                sym = chunk[0]
                sub = df.dropna(subset=["Close"])
                if len(sub) >= 60:
                    history_map[sym] = sub
            else:
                for sym in chunk:
                    if sym in df.columns.get_level_values(0):
                        sub = df[sym].dropna(subset=["Close"])
                        if len(sub) >= 60:
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
        return None

    price = float(close.iloc[-1])
    high_52w = float(close.max())
    dist_52w = ((price - high_52w) / high_52w) * 100.0 if high_52w > 0 else 0.0

    price_3m_ago = float(close.iloc[-63]) if len(close) >= 63 else float(close.iloc[0])
    ret_3m = ((price - price_3m_ago) / price_3m_ago) * 100.0 if price_3m_ago > 0 else 0.0
    rs_3m = ret_3m - sp500_3m_ret

    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
    rsi = compute_rsi(close, 14)

    avg_vol_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else 0.0
    dollar_vol = avg_vol_20 * price

    # Tech-Score (0-100)
    tech_score = 0
    if price > sma50:
        tech_score += 25
    if price > sma200:
        tech_score += 25
    if sma50 > sma200:
        tech_score += 10
    if rs_3m > 0:
        tech_score += min(20, int(rs_3m * 1.2))
    if 45 <= rsi <= 68:
        tech_score += 10
    elif 30 <= rsi < 45:
        tech_score += 5
    if dist_52w >= -15:
        tech_score += 10
    tech_score = max(0, min(100, tech_score))

    # Funda-Score (0-100)
    funda_score = 40  # Solider Basiswert, falls yfinance einzelne Felder nicht liefert
    rev_growth = info.get("revenueGrowth")
    roe = info.get("returnOnEquity")
    profit_margin = info.get("profitMargins")
    fwd_pe = info.get("forwardPE")
    debt_eq = info.get("debtToEquity")

    if isinstance(rev_growth, (int, float)):
        if rev_growth > 0.20:
            funda_score += 20
        elif rev_growth > 0.08:
            funda_score += 12
        elif rev_growth > 0:
            funda_score += 5
        else:
            funda_score -= 10

    if isinstance(roe, (int, float)):
        if roe > 0.18:
            funda_score += 15
        elif roe > 0.08:
            funda_score += 8
        elif roe < 0:
            funda_score -= 10

    if isinstance(profit_margin, (int, float)):
        if profit_margin > 0.15:
            funda_score += 15
        elif profit_margin > 0.05:
            funda_score += 8
        elif profit_margin < 0:
            funda_score -= 8

    if isinstance(fwd_pe, (int, float)) and 0 < fwd_pe < 25:
        funda_score += 10

    if isinstance(debt_eq, (int, float)) and debt_eq > 200:
        funda_score -= 10

    funda_score = max(0, min(100, funda_score))

    total_score = round(tech_score * 0.6 + funda_score * 0.4, 1)

    # Phase bestimmen
    if price > sma50 and price > sma200 and rs_3m > 0:
        phase = "Aufwärtstrend"
    elif price > sma50 and price <= sma200:
        phase = "Turnaround"
    elif price <= sma50 and price > sma200:
        phase = "Rücksetzer"
    else:
        phase = "Boden abwarten"

    # Signal bestimmen
    if total_score >= 62 and price > sma50:
        signal = "kaufen"
    elif total_score >= 45:
        signal = "halten"
    else:
        signal = "meiden"

    chance = max(1, min(5, math.ceil(total_score / 20)))

    # Risiko (1-5)
    daily_vol = float(close.pct_change().dropna().std() * np.sqrt(252) * 100)
    if daily_vol < 25:
        risk = 2
    elif daily_vol < 40:
        risk = 3
    elif daily_vol < 60:
        risk = 4
    else:
        risk = 5

    name = info.get("shortName") or info.get("longName") or sym
    currency = info.get("currency") or ("EUR" if sym.endswith((".DE", ".PA")) else "USD")
    sector = info.get("sector") or "Sonstige"

    return {
        "ticker": sym,
        "name": name,
        "is_core": sym in core_set,
        "sector": sector,
        "currency": currency,
        "price": round(price, 2),
        "signal": signal,
        "phase": phase,
        "chance": chance,
        "risk": risk,
        "tech_score": tech_score,
        "funda_score": funda_score,
        "total_score": total_score,
        "ret_3m": round(ret_3m, 1),
        "rs_3m": round(rs_3m, 1),
        "dist_52w": round(dist_52w, 1),
        "rsi": rsi,
        "avg_volume": int(avg_vol_20),
        "dollar_volume": int(dollar_vol),
        "above_sma50": bool(price > sma50),
        "above_sma200": bool(price > sma200),
    }


def main():
    tickers = build_ticker_universe()
    core_set = set(CORE_UNIVERSE)

    history_map = download_history_batches(tickers)
    print(f"Gültige Kursreihen geladen: {len(history_map)}")

    sp500_hist = history_map.get("^GSPC")
    if sp500_hist is not None and len(sp500_hist["Close"]) >= 63:
        sp_c = sp500_hist["Close"].dropna()
        sp500_3m_ret = ((float(sp_c.iloc[-1]) - float(sp_c.iloc[-63])) / float(sp_c.iloc[-63])) * 100.0
    else:
        sp500_3m_ret = 0.0

    valid_symbols = [s for s in tickers if s in history_map]

    info_map = {}
    with ThreadPoolExecutor(max_workers=INFO_WORKERS) as executor:
        futures = {executor.submit(fetch_info_single, sym): sym for sym in valid_symbols}
        for idx, future in enumerate(as_completed(futures), 1):
            sym, info = future.result()
            info_map[sym] = info
            if idx % 100 == 0:
                print(f"Fundamentaldaten geladen: {idx}/{len(valid_symbols)}")

    results = []
    for sym in valid_symbols:
        item = score_stock(sym, history_map[sym], info_map.get(sym, {}), sp500_3m_ret, core_set)
        if item is not None:
            results.append(item)

    results.sort(key=lambda x: (x["is_core"], x["total_score"]), reverse=True)

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "count": len(results),
        "stocks": results,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Fertig! {len(results)} Aktien in data.json gespeichert.")


if __name__ == "__main__":
    main()
