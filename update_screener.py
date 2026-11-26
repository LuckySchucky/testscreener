import json
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# Erweiterte Watchlist & Fokus-Sektoren
UNIVERSE = [
    # AI & Cloud Infrastruktur (2030 Fokus)
    {"ticker": "ORCL", "strategy": "Wachstum", "tag": "AI & Cloud 2030"},
    {"ticker": "IREN", "strategy": "Turnaround", "tag": "AI Data Center"},
    {"ticker": "CIFR", "strategy": "Turnaround", "tag": "AI Data Center"},
    {"ticker": "DELL", "strategy": "Wachstum", "tag": "AI & Cloud 2030"},
    {"ticker": "HPE", "strategy": "Cashflow", "tag": "AI Data Center"},
    {"ticker": "SMCI", "strategy": "Turnaround", "tag": "AI Data Center"},
    {"ticker": "NVDA", "strategy": "Wachstum", "tag": "AI & Cloud 2030"},
    {"ticker": "MSFT", "strategy": "Wachstum", "tag": "AI & Cloud 2030"},
    {"ticker": "AAPL", "strategy": "Cashflow", "tag": "Big Tech"},

    # Halbleiter & Photonics
    {"ticker": "VLN", "strategy": "Turnaround", "tag": "Semiconductors"},
    {"ticker": "INTC", "strategy": "Turnaround", "tag": "Semiconductors"},
    {"ticker": "AMD", "strategy": "Wachstum", "tag": "Semiconductors"},
    {"ticker": "MU", "strategy": "Wachstum", "tag": "Semiconductors"},
    {"ticker": "ARM", "strategy": "Wachstum", "tag": "Semiconductors"},
    {"ticker": "ASML", "strategy": "Wachstum", "tag": "Semiconductors"},

    # Energie, Speicher & Grid-Engpässe
    {"ticker": "EOSE", "strategy": "Turnaround", "tag": "Energy & Grid"},
    {"ticker": "RUN", "strategy": "Turnaround", "tag": "Energy & Grid"},
    {"ticker": "FLNC", "strategy": "Wachstum", "tag": "Energy & Grid"},
    {"ticker": "DFLI", "strategy": "Turnaround", "tag": "Energy & Grid"},
    {"ticker": "NVVE", "strategy": "Turnaround", "tag": "Energy & Grid"},
    {"ticker": "BTU", "strategy": "Cashflow", "tag": "Energy & Grid"},
    {"ticker": "VST", "strategy": "Wachstum", "tag": "Energy & Grid"},
    {"ticker": "CEG", "strategy": "Wachstum", "tag": "Energy & Grid"},

    # Small Caps / Drohnen / Spezialwerte
    {"ticker": "ZENA", "strategy": "Wachstum", "tag": "Tech & Robotics"},

    # Healthcare & Biotech
    {"ticker": "HIMS", "strategy": "Wachstum", "tag": "Healthcare"},
    {"ticker": "NVO", "strategy": "Wachstum", "tag": "Healthcare"},
    {"ticker": "LLY", "strategy": "Wachstum", "tag": "Healthcare"},

    # Europa & Turnaround-Klassiker
    {"ticker": "RHM.DE", "strategy": "Wachstum", "tag": "Europa & Industrie"},
    {"ticker": "SIE.DE", "strategy": "Dividende", "tag": "Europa & Industrie"},
    {"ticker": "BAS.DE", "strategy": "Dividende", "tag": "Turnaround Watch"},
    {"ticker": "BAYN.DE", "strategy": "Turnaround", "tag": "Turnaround Watch"},
    {"ticker": "PYPL", "strategy": "Turnaround", "tag": "Turnaround Watch"},
    {"ticker": "NKE", "strategy": "Turnaround", "tag": "Turnaround Watch"},

    # Asien / China Tech & Speicherchips
    {"ticker": "BABA", "strategy": "Turnaround", "tag": "Asien & China Tech"},
    {"ticker": "JD", "strategy": "Turnaround", "tag": "Asien & China Tech"},
    {"ticker": "BIDU", "strategy": "Turnaround", "tag": "Asien & China Tech"},
    {"ticker": "005930.KS", "strategy": "Cashflow", "tag": "Asien & China Tech"},
    {"ticker": "000660.KS", "strategy": "Wachstum", "tag": "Asien & China Tech"},
]

COUNTRY_MAP = {
    "United States": "USA",
    "Germany": "Deutschland",
    "Netherlands": "Niederlande",
    "United Kingdom": "Großbritannien",
    "Denmark": "Dänemark",
    "Israel": "Israel",
    "Australia": "Australien",
    "China": "China",
    "Hong Kong": "Hongkong",
    "South Korea": "Südkorea",
    "Canada": "Kanada",
}

SECTOR_MAP = {
    "Technology": "Informationstechnologie",
    "Healthcare": "Gesundheitswesen",
    "Industrials": "Industrie",
    "Consumer Cyclical": "Nicht-Basiskonsumgüter",
    "Consumer Defensive": "Basiskonsumgüter",
    "Financial Services": "Finanzen",
    "Energy": "Energie",
    "Utilities": "Versorger",
    "Basic Materials": "Grundstoffe",
    "Communication Services": "Kommunikation",
}

def clamp(val, low=0, high=100):
    return int(round(max(low, min(high, val))))

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def calc_tech_score(close: pd.Series):
    """Technisches Rating (0-100): Trendlage, RSI-Zone, Abstand 52W-Hoch, Momentum."""
    price = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
    high_52w = float(close.tail(252).max())
    low_52w = float(close.tail(252).min())
    rsi = calc_rsi(close)
    mom_3m = (price / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else 0.0

    # 1. Trendlage (max 35 Punkte)
    trend_score = 0
    if price > sma200:
        trend_score += 20
    if price > sma50:
        trend_score += 10
    if sma50 > sma200:
        trend_score += 5

    # 2. RSI-Zone (max 25 Punkte)
    if 42 <= rsi <= 65:
        rsi_score = 25
    elif 30 <= rsi < 42:
        rsi_score = 20
    elif rsi < 30:
        rsi_score = 15
    elif 65 < rsi <= 75:
        rsi_score = 14
    else:
        rsi_score = 6

    # 3. Abstand zum 52W-Hoch (max 20 Punkte)
    dist_high = (price / high_52w) if high_52w > 0 else 0.5
    high_score = dist_high * 20

    # 4. 3-Monats-Momentum (max 20 Punkte)
    mom_score = max(0, min(20, 10 + mom_3m * 0.5))

    score = clamp(trend_score + rsi_score + high_score + mom_score)
    metrics = {
        "price": round(price, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "high52w": round(high_52w, 2),
        "low52w": round(low_52w, 2),
        "rsi": round(rsi, 1),
        "mom3m": round(mom_3m, 1),
    }
    return score, metrics

def calc_funda_score(info: dict):
    """Fundamentales Rating (0-100): Umsatzwachstum, ROE, Verschuldung, Bewertung."""
    rev_growth = info.get("revenueGrowth")
    roe = info.get("returnOnEquity")
    debt_to_eq = info.get("debtToEquity")
    forward_pe = info.get("forwardPE")
    profit_margin = info.get("profitMargins")

    # 1. Umsatzwachstum (max 30 Punkte)
    if rev_growth is None:
        g_score = 15
    else:
        g_score = max(0, min(30, 12 + rev_growth * 80))

    # 2. Eigenkapitalrendite / Profitabilität (max 30 Punkte)
    if roe is not None:
        roe_score = max(0, min(30, roe * 100))
    elif profit_margin is not None:
        roe_score = max(0, min(30, 12 + profit_margin * 60))
    else:
        roe_score = 14

    # 3. Verschuldungsgrad (max 20 Punkte)
    if debt_to_eq is None:
        d_score = 12
    elif debt_to_eq < 50:
        d_score = 20
    elif debt_to_eq < 120:
        d_score = 14
    elif debt_to_eq < 250:
        d_score = 8
    else:
        d_score = 3

    # 4. Bewertung / Forward KGV (max 20 Punkte)
    if forward_pe is None or forward_pe <= 0:
        pe_score = 8
    elif forward_pe < 18:
        pe_score = 20
    elif forward_pe < 28:
        pe_score = 15
    elif forward_pe < 45:
        pe_score = 9
    else:
        pe_score = 4

    return clamp(g_score + roe_score + d_score + pe_score)

def determine_status_and_rating(m: dict, tech_score: int, funda_score: int):
    """Automatische Zuweisung von Chart-Phase & Signal anhand Trendlinien & Unterstützungszonen."""
    p = m["price"]
    s50 = m["sma50"]
    s200 = m["sma200"]
    rsi = m["rsi"]
    low52 = m["low52w"]
    high52 = m["high52w"]
    pos_in_range = (p - low52) / max(0.01, (high52 - low52))
    total_score = tech_score * 0.55 + funda_score * 0.45

    if p < s200 * 0.94 and p < s50 and rsi < 38:
        if pos_in_range < 0.22:
            status = "Bodenbildung"
        else:
            status = "Trendbruch"
    elif p < s200 and p > s50 and rsi >= 45:
        status = "Turnaround"
    elif p < s200 and pos_in_range < 0.25:
        status = "Boden abwarten"
    elif p > s200 and p < s50:
        status = "Dynamikverlust"
    elif p > s200 and rsi > 72:
        status = "Regression abwarten"
    elif p > s200 and p > s50 and total_score >= 62:
        status = "Position verwalten"
    else:
        status = "Underperformance"

    if total_score >= 60 and status in ("Bodenbildung", "Turnaround", "Position verwalten", "Regression abwarten"):
        rating = "kaufen"
    elif total_score < 42 or status == "Trendbruch":
        rating = "verkaufen"
    else:
        rating = "halten"

    chance = max(1, min(5, math.ceil(total_score / 20)))
    vol_spread = (high52 - low52) / max(0.01, low52)
    risk = 5 if vol_spread > 1.2 or funda_score < 40 else (4 if vol_spread > 0.75 else (3 if vol_spread > 0.45 else 2))

    return status, rating, chance, risk

def build_dataset():
    results = []
    for item in UNIVERSE:
        symbol = item["ticker"]
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1y", interval="1d")
            if hist.empty or len(hist) < 30:
                continue

            close = hist["Close"].dropna()
            info = tk.info or {}

            tech_score, m = calc_tech_score(close)
            funda_score = calc_funda_score(info)
            status, rating, chance, risk = determine_status_and_rating(m, tech_score, funda_score)

            step = max(1, len(close) // 24)
            chart_points = [round(float(x), 2) for x in close.iloc[::step].tolist()]
            if chart_points[-1] != m["price"]:
                chart_points.append(m["price"])

            raw_country = info.get("country", "United States")
            raw_sector = info.get("sector", "Technology")

            results.append({
                "name": info.get("shortName") or info.get("longName") or symbol,
                "ticker": symbol,
                "isin": info.get("isin") or "—",
                "strategy": item["strategy"],
                "tag": item["tag"],
                "chance": chance,
                "risk": risk,
                "rating": rating,
                "status": status,
                "country": COUNTRY_MAP.get(raw_country, raw_country),
                "sector": SECTOR_MAP.get(raw_sector, raw_sector),
                "tech_score": tech_score,
                "funda_score": funda_score,
                "price": m["price"],
                "currency": info.get("currency", "USD"),
                "sma50": m["sma50"],
                "sma200": m["sma200"],
                "rsi": m["rsi"],
                "mom3m": m["mom3m"],
                "forwardPE": round(info["forwardPE"], 1) if info.get("forwardPE") else None,
                "revGrowth": round(info["revenueGrowth"] * 100, 1) if info.get("revenueGrowth") else None,
                "history": chart_points,
            })
            print(f"OK: {symbol} -> {rating} | {status} | Tech={tech_score}, Funda={funda_score}")
        except Exception as e:
            print(f"Fehler bei {symbol}: {e}")

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC"),
        "count": len(results),
        "stocks": results,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_dataset()
