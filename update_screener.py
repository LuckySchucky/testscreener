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

MAX_TICKERS = 1000
BATCH_SIZE = 125
INFO_WORKERS = 6

CORE_UNIVERSE = [
    "ORCL", "IREN", "ZENA", "VLN", "RUN", "EOSE", "DFLI", "BTU", "NVVE",
    "DELL", "INTC", "AAPL", "VRT", "CLS", "CRWD", "LLY", "NVDA", "MU",
    "COHR", "AMBA", "ENR.DE", "RHM.DE", "SU.PA", "005930.KS", "000660.KS"
]

EXTRA_UNIVERSE = [
    "NBIS", "HIMS", "RKLB", "RDW", "BTDR", "CLSK", "MARA", "RIOT", "SMCI",
    "ARM", "MRVL", "AVGO", "TSM", "ASML", "SMH", "SOXL", "PLTR", "APP",
    "SNOW", "DDOG", "NET", "MDB", "COIN", "HOOD", "DKNG", "SOFI", "UPST",
    "AFRM", "CELH", "ONON", "DECK", "CAVA", "DUOL", "UBER", "ABNB", "DASH",
    "SHOP", "MELI", "NU", "SE", "GRAB", "BABA", "PDD", "JD", "BIDU",
    "BE", "HPE", "ZM", "BBY", "MLTX", "BBIO", "VRTX", "CHTR", "VKTX",
    "NVO", "SNDK", "TE", "CISS", "FTK", "METC", "KSS", "TMDX", "OKLO",
    "SMR", "CCJ", "UEC", "UUUU", "LEU", "VST", "CEG", "NRG", "TLN",
    "EQNR", "SHEL", "TTE", "BP", "XOM", "CVX", "OXY", "SLB", "HAL",
    "SIE.DE", "SAP.DE", "ALV.DE", "MUV2.DE", "DTE.DE", "IFX.DE", "MBG.DE",
    "BMW.DE", "VOW3.DE", "BAS.DE", "BAYN.DE", "AIR.PA", "MC.PA", "OR.PA",
    "SAN.PA", "RMS.PA", "KER.PA", "AI.PA", "DG.PA", "BNP.PA", "CS.PA",
    "HAG.DE", "NDA.DE", "AFX.DE", "EVD.DE", "KRN.DE", "LEG.DE", "P911.DE",
    "DHER.DE", "HFG.DE", "TMV.DE", "SGL.DE", "WAF.DE", "SZG.DE", "G1A.DE"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def fetch_wiki_tables(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=25)
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
                    sym = sym.split(":")[-1].strip().replace(".DE", "") + ".DE"
                elif suffix == ".PA":
                    sym = sym.split(":")[-1].strip().replace(".PA", "") + ".PA"
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
    russell1000 = extract_symbols_from_tables("https://en.wikipedia.org/wiki/Russell_1000_Index")
    dax = extract_symbols_from_tables("https://en.wikipedia.org/wiki/DAX", suffix=".DE")
    mdax = extract_symbols_from_tables("https://en.wikipedia.org/wiki/MDAX", suffix=".DE")
    sdax = extract_symbols_from_tables("https://en.wikipedia.org/wiki/SDAX", suffix=".DE")
    cac40 = extract_symbols_from_tables("https://en.wikipedia.org/wiki/CAC_40", suffix=".PA")

    combined = []
    seen = set()

    for sym in CORE_UNIVERSE + sp500 + nasdaq100 + russell1000 + dax + mdax + sdax + cac40 + EXTRA_UNIVERSE:
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
        time.sleep(0.9)

    return history_map


def fetch_info_single(sym: str) -> tuple[str, dict]:
    time.sleep(random.uniform(0.08, 0.25))
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
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
    rsi = compute_rsi(close, 14)

    avg_vol_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else 0.0
    last_vol = float(volume.iloc[-1]) if len(volume) >= 1 else 0.0
    vol_spike = (last_vol > avg_vol_20 * 1.35) if avg_vol_20 > 0 else False

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

    rev_growth = info.get("revenueGrowth") if info else None
    roe = info.get("returnOnEquity") if info else None
    profit_margin = info.get("profitMargins") if info else None
    fwd_pe = info.get("forwardPE") if info else None
    debt_eq = info.get("debtToEquity") if info else None
    div_yield = info.get("dividendYield") if info else None

    if not info:
        funda_score = 50
    else:
        funda_score = 45
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

    if price > sma50 and price > sma200 and rs_3m > 5 and rsi <= 72:
        status = "Starker Aufwärtstrend"
    elif price > sma50 and price <= sma200:
        status = "Erster Ausbruch"
    elif price <= sma50 and price > sma200 and rsi < 45:
        status = "Gesunder Rücksetzer"
    elif price <= sma50 and price > sma200:
        status = "Trend-Abkühlung"
    elif price < sma200 and dist_52w < -25 and rsi >= 42 and price > sma20:
        status = "Boden-Check"
    elif price < sma50 and price < sma200 and dist_52w < -20:
        status = "Korrektur / Abwärtstrend"
    elif price < sma200 and rs_3m < -10:
        status = "Klarer Abwärtstrend"
    else:
        status = "Seitwärts / Schwach"

    if isinstance(div_yield, (int, float)) and div_yield >= 0.032:
        strategy = "Dividende"
    elif dist_52w <= -25 and funda_score >= 55:
        strategy = "Turnaround"
    elif isinstance(profit_margin, (int, float)) and profit_margin >= 0.18 and (not isinstance(rev_growth, (int, float)) or rev_growth < 0.12):
        strategy = "Cashflow"
    else:
        strategy = "Wachstum"

    # Erweiterte Kaufsignal-Logik für Swing- & Rebound-Setups
    is_trend_buy = (total_score >= 56 and price > sma50 and status == "Starker Aufwärtstrend")
    is_breakout_buy = (total_score >= 55 and price > sma50 and status == "Erster Ausbruch" and (rs_3m > -3 or vol_spike))
    is_pullback_buy = (status == "Gesunder Rücksetzer" and price > sma200 and funda_score >= 60 and rsi >= 32)
    is_bottom_buy = (status == "Boden-Check" and total_score >= 52 and (rs_3m > -5 or vol_spike))

    if is_trend_buy or is_breakout_buy or is_pullback_buy or is_bottom_buy:
        rating = "kaufen"
    elif total_score < 38 or status == "Klarer Abwärtstrend":
        rating = "verkaufen"
    else:
        rating = "halten"

    if rating == "kaufen":
        if is_pullback_buy:
            setup_label = "Pullback-Chance"
        elif is_breakout_buy:
            setup_label = "Ausbruch-Setup"
        elif is_bottom_buy:
            setup_label = "Boden-Rebound"
        else:
            setup_label = "Trend-Fortsetzung"
    elif rating == "verkaufen":
        setup_label = "Schwacher Trend"
    else:
        setup_label = "Beobachten"

    chance = max(1, min(5, math.ceil(total_score / 20)))

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
    country = info.get("country") or ("Germany" if sym.endswith(".DE") else ("France" if sym.endswith(".PA") else "USA"))
    isin = info.get("isin") or "—"
    tag = "Fokus-Liste" if sym in core_set else setup_label

    history_pts = [round(float(x), 2) for x in close.tail(60).tolist()]

    return {
        "ticker": sym,
        "name": name,
        "isin": isin,
        "tag": tag,
        "setup_label": setup_label,
        "strategy": strategy,
        "rating": rating,
        "status": status,
        "country": country,
        "sector": sector,
        "currency": currency,
        "price": round(price, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "high52w": round(high_52w, 2),
        "distHighPct": round(dist_52w, 1),
        "chance": chance,
        "risk": risk,
        "tech_score": tech_score,
        "funda_score": funda_score,
        "total_score": total_score,
        "mom3m": round(ret_3m, 1),
        "rs3m": round(rs_3m, 1),
        "rsi": rsi,
        "avgVolume": int(avg_vol_20),
        "forwardPE": round(float(fwd_pe), 1) if isinstance(fwd_pe, (int, float)) and fwd_pe > 0 else "—",
        "revGrowth": round(float(rev_growth) * 100, 1) if isinstance(rev_growth, (int, float)) else None,
        "history": history_pts,
    }


def main():
    tickers = build_ticker_universe()
    core_set = set(CORE_UNIVERSE)

    history_map = download_history_batches(tickers)
    print(f"Gültige Kursreihen geladen: {len(history_map)} / {len(tickers)}")

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

    results.sort(key=lambda x: (x["tag"] == "Fokus-Liste", x["total_score"]), reverse=True)

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "benchmark_mom3m": round(sp500_3m_ret, 1),
        "count": len(results),
        "stocks": results,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Fertig! {len(results)} Aktien in data.json gespeichert.")


if __name__ == "__main__":
    main()
