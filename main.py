import os
import json
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys

import pytz
import numpy as np
import pandas as pd
import yfinance as yf

from ta.trend import (
    SMAIndicator, EMAIndicator, WMAIndicator,
    MACD, ADXIndicator, AroonIndicator, CCIIndicator, DPOIndicator,
    MassIndex, IchimokuIndicator, PSARIndicator, STCIndicator,
    TRIXIndicator, VortexIndicator,
)
from ta.volatility import (
    KeltnerChannel, DonchianChannel, AverageTrueRange, BollingerBands, UlcerIndex,
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator,
    AwesomeOscillatorIndicator, KAMAIndicator,
    PercentagePriceOscillator, PercentageVolumeOscillator,
    TSIIndicator, UltimateOscillator,
)
from ta.volume import (
    OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator,
    AccDistIndexIndicator, MFIIndicator, ForceIndexIndicator,
    EaseOfMovementIndicator, VolumePriceTrendIndicator,
    NegativeVolumeIndexIndicator, VolumeWeightedAveragePrice,
)

# ── Config ─────────────────────────────────────────────────────────────────────
DB_NAME      = "SmallCap_Technical.db"
README_FILE  = "README.md"
SYMBOLS_FILE = "symbols.json"
IST          = pytz.timezone("Asia/Kolkata")

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_fh = RotatingFileHandler("data_fetch.log", maxBytes=5 * 1024 * 1024, backupCount=5)
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_fh)
_sh = logging.StreamHandler(sys.stdout)
_sh.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
logger.addHandler(_sh)

# ── Schema ─────────────────────────────────────────────────────────────────────
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS technical_indicators (
    datetime TEXT,
    stock_name TEXT,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    sma_5 REAL, sma_10 REAL, sma_20 REAL, sma_50 REAL, sma_100 REAL, sma_200 REAL,
    ema_5 REAL, ema_10 REAL, ema_20 REAL, ema_50 REAL, ema_100 REAL, ema_200 REAL,
    wma_10 REAL, wma_20 REAL,
    macd REAL, macd_signal REAL, macd_diff REAL,
    adx REAL, adx_pos REAL, adx_neg REAL,
    aroon_up REAL, aroon_down REAL, aroon_indicator REAL,
    cci REAL, dpo REAL, mass_index REAL,
    ichimoku_a REAL, ichimoku_b REAL, ichimoku_base REAL, ichimoku_conv REAL,
    psar REAL, stc REAL, trix REAL,
    vortex_pos REAL, vortex_neg REAL,
    kc_upper REAL, kc_middle REAL, kc_lower REAL,
    dc_upper REAL, dc_middle REAL, dc_lower REAL,
    atr REAL,
    bb_upper REAL, bb_middle REAL, bb_lower REAL, bb_pband REAL, bb_wband REAL,
    ulcer_index REAL,
    rsi_7 REAL, rsi_14 REAL, rsi_21 REAL,
    stoch_k REAL, stoch_d REAL,
    roc REAL, williams_r REAL,
    awesome_oscillator REAL, kama REAL,
    ppo REAL, pvo REAL, tsi REAL, ultimate_oscillator REAL,
    obv REAL, cmf REAL, acc_dist REAL, mfi REAL,
    force_index REAL, eom REAL, vpt REAL, nvi REAL, vwap REAL,
    price_change_pct REAL,
    signal TEXT,
    updated_at TEXT,
    PRIMARY KEY (datetime, stock_name)
)
"""

# ── DB helpers ─────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(CREATE_SQL)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(technical_indicators)").fetchall()]
        if "updated_at" not in cols:
            conn.execute("ALTER TABLE technical_indicators ADD COLUMN updated_at TEXT")
        # Remove stale daily candles (stored as HH:MM:SS = 00:00:00 from 1d interval)
        conn.execute("DELETE FROM technical_indicators WHERE datetime LIKE '%00:00:00'")
        conn.commit()


def upsert_df(df_rows: pd.DataFrame):
    if df_rows.empty:
        return
    cols = list(df_rows.columns)
    ph   = ", ".join("?" * len(cols))
    sql  = f"INSERT OR REPLACE INTO technical_indicators ({', '.join(cols)}) VALUES ({ph})"
    data = [
        [None if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in row]
        for row in df_rows.values.tolist()
    ]
    with sqlite3.connect(DB_NAME) as conn:
        conn.executemany(sql, data)
        conn.commit()


# ── Fetch ──────────────────────────────────────────────────────────────────────
MARKET_OPEN  = datetime.strptime("09:15", "%H:%M").time()
MARKET_CLOSE = datetime.strptime("15:30", "%H:%M").time()

def fetch(symbol: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(symbol, interval="1m", period="5d", progress=False, auto_adjust=True)
        if raw.empty:
            logger.warning(f"No data: {symbol}")
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        raw = raw.rename(columns={"adj_close": "close"})
        raw.index = pd.to_datetime(raw.index)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert(IST)
        df = raw[["open", "high", "low", "close", "volume"]].copy()
        df = df[df.index.day_of_week < 5]  # Mon–Fri only
        df = df.between_time("09:15", "15:30")
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        return df
    except Exception as e:
        logger.error(f"Fetch error {symbol}: {e}")
        return None


# ── Signal ─────────────────────────────────────────────────────────────────────
def _signal(row: pd.Series) -> str:
    try:
        c, e20, r, m, ms, adx = (
            row["close"], row["ema_20"], row["rsi_14"],
            row["macd"],  row["macd_signal"], row["adx"],
        )
        if any(pd.isna(x) for x in [c, e20, r, m, ms, adx]):
            return "HOLD"
        if c > e20 and r > 50 and m > ms and adx > 20:
            return "BUY"
        if c < e20 and r < 50 and m < ms and adx > 20:
            return "SELL"
    except Exception:
        pass
    return "HOLD"


# ── Indicators (vectorized) ────────────────────────────────────────────────────
def build_indicator_df(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    has_vol = v.replace(0, np.nan).notna().sum() > 20

    out = pd.DataFrame(index=df.index)
    out["datetime"]   = df.index.strftime("%Y-%m-%d %H:%M:%S")
    out["stock_name"] = symbol
    out["open"]       = o.values
    out["high"]       = h.values
    out["low"]        = l.values
    out["close"]      = c.values
    out["volume"]     = v.values

    # SMA
    for p in [5, 10, 20, 50, 100, 200]:
        out[f"sma_{p}"] = SMAIndicator(c, window=p, fillna=False).sma_indicator()

    # EMA
    for p in [5, 10, 20, 50, 100, 200]:
        out[f"ema_{p}"] = EMAIndicator(c, window=p, fillna=False).ema_indicator()

    # WMA
    for p in [10, 20]:
        out[f"wma_{p}"] = WMAIndicator(c, window=p, fillna=False).wma()

    # MACD
    _macd = MACD(c, fillna=False)
    out["macd"]        = _macd.macd()
    out["macd_signal"] = _macd.macd_signal()
    out["macd_diff"]   = _macd.macd_diff()

    # ADX
    _adx = ADXIndicator(h, l, c, fillna=False)
    out["adx"]     = _adx.adx()
    out["adx_pos"] = _adx.adx_pos()
    out["adx_neg"] = _adx.adx_neg()

    # Aroon
    _ar = AroonIndicator(h, l, fillna=False)
    out["aroon_up"]        = _ar.aroon_up()
    out["aroon_down"]      = _ar.aroon_down()
    out["aroon_indicator"] = _ar.aroon_indicator()

    # CCI / DPO / Mass Index
    out["cci"]        = CCIIndicator(h, l, c, fillna=False).cci()
    out["dpo"]        = DPOIndicator(c, fillna=False).dpo()
    out["mass_index"] = MassIndex(h, l, fillna=False).mass_index()

    # Ichimoku
    _ich = IchimokuIndicator(h, l, fillna=False)
    out["ichimoku_a"]    = _ich.ichimoku_a()
    out["ichimoku_b"]    = _ich.ichimoku_b()
    out["ichimoku_base"] = _ich.ichimoku_base_line()
    out["ichimoku_conv"] = _ich.ichimoku_conversion_line()

    # PSAR / STC / TRIX
    out["psar"] = PSARIndicator(h, l, c, fillna=False).psar()
    out["stc"]  = STCIndicator(c, fillna=False).stc()
    out["trix"] = TRIXIndicator(c, fillna=False).trix()

    # Vortex
    _vi = VortexIndicator(h, l, c, fillna=False)
    out["vortex_pos"] = _vi.vortex_indicator_pos()
    out["vortex_neg"] = _vi.vortex_indicator_neg()

    # Keltner
    _kc = KeltnerChannel(h, l, c, fillna=False)
    out["kc_upper"]  = _kc.keltner_channel_hband()
    out["kc_middle"] = _kc.keltner_channel_mband()
    out["kc_lower"]  = _kc.keltner_channel_lband()

    # Donchian
    _dc = DonchianChannel(h, l, c, fillna=False)
    out["dc_upper"]  = _dc.donchian_channel_hband()
    out["dc_middle"] = _dc.donchian_channel_mband()
    out["dc_lower"]  = _dc.donchian_channel_lband()

    # ATR / Bollinger / Ulcer
    out["atr"] = AverageTrueRange(h, l, c, fillna=False).average_true_range()
    _bb = BollingerBands(c, fillna=False)
    out["bb_upper"]    = _bb.bollinger_hband()
    out["bb_middle"]   = _bb.bollinger_mavg()
    out["bb_lower"]    = _bb.bollinger_lband()
    out["bb_pband"]    = _bb.bollinger_pband()
    out["bb_wband"]    = _bb.bollinger_wband()
    out["ulcer_index"] = UlcerIndex(c, fillna=False).ulcer_index()

    # RSI
    for p in [7, 14, 21]:
        out[f"rsi_{p}"] = RSIIndicator(c, window=p, fillna=False).rsi()

    # Stochastic
    _st = StochasticOscillator(h, l, c, fillna=False)
    out["stoch_k"] = _st.stoch()
    out["stoch_d"] = _st.stoch_signal()

    # ROC / Williams %R / AO / KAMA
    out["roc"]                = ROCIndicator(c, fillna=False).roc()
    out["williams_r"]         = WilliamsRIndicator(h, l, c, fillna=False).williams_r()
    out["awesome_oscillator"] = AwesomeOscillatorIndicator(h, l, fillna=False).awesome_oscillator()
    out["kama"]               = KAMAIndicator(c, fillna=False).kama()

    # PPO / PVO / TSI / Ultimate Oscillator
    out["ppo"] = PercentagePriceOscillator(c, fillna=False).ppo()
    out["pvo"] = PercentageVolumeOscillator(v, fillna=False).pvo() if has_vol else np.nan
    out["tsi"] = TSIIndicator(c, fillna=False).tsi()
    out["ultimate_oscillator"] = UltimateOscillator(h, l, c, fillna=False).ultimate_oscillator()

    # Volume indicators (NULL for indexes without volume)
    if has_vol:
        out["obv"]         = OnBalanceVolumeIndicator(c, v, fillna=False).on_balance_volume()
        out["cmf"]         = ChaikinMoneyFlowIndicator(h, l, c, v, fillna=False).chaikin_money_flow()
        out["acc_dist"]    = AccDistIndexIndicator(h, l, c, v, fillna=False).acc_dist_index()
        out["mfi"]         = MFIIndicator(h, l, c, v, fillna=False).money_flow_index()
        out["force_index"] = ForceIndexIndicator(c, v, fillna=False).force_index()
        out["eom"]         = EaseOfMovementIndicator(h, l, v, fillna=False).ease_of_movement()
        out["vpt"]         = VolumePriceTrendIndicator(c, v, fillna=False).volume_price_trend()
        out["nvi"]         = NegativeVolumeIndexIndicator(c, v, fillna=False).negative_volume_index()
        try:
            out["vwap"] = VolumeWeightedAveragePrice(h, l, c, v, fillna=False).volume_weighted_average_price()
        except Exception:
            out["vwap"] = np.nan
    else:
        for col in ["obv", "cmf", "acc_dist", "mfi", "force_index", "eom", "vpt", "nvi", "vwap"]:
            out[col] = np.nan

    out["price_change_pct"] = c.pct_change() * 100
    out["signal"]           = out.apply(_signal, axis=1)
    out["updated_at"]       = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    return out.reset_index(drop=True)


# ── README helpers ─────────────────────────────────────────────────────────────
def _dn(sym: str) -> str:
    return sym.replace(".", "_")


def _fmt(val, dec=2):
    try:
        f = float(val)
        return "—" if np.isnan(f) else f"{f:.{dec}f}"
    except Exception:
        return "—"


def _table3(lines, rows):
    lines.append("| Indicator | Value | Indicator | Value | Indicator | Value |\n")
    lines.append("|-----------|------:|-----------|------:|-----------|------:|\n")
    while len(rows) % 3:
        rows.append(("", ""))
    for i in range(0, len(rows), 3):
        a, b, c = rows[i], rows[i+1], rows[i+2]
        lines.append(f"| {a[0]} | {a[1]} | {b[0]} | {b[1]} | {c[0]} | {c[1]} |\n")


# ── README ─────────────────────────────────────────────────────────────────────
INDEX_DISPLAY = {
    "^NSEI":      "Nifty 50",
    "^BSESN":     "Sensex",
    "^NSEBANK":   "BankNifty",
    "^NSEMDCP50": "SmallcapNifty",
}


def update_readme(all_symbols: list[str]):
    conn = sqlite3.connect(DB_NAME)

    latest_rows = {}
    for sym in all_symbols:
        try:
            r = pd.read_sql_query(
                "SELECT * FROM technical_indicators WHERE stock_name=? ORDER BY datetime DESC LIMIT 1",
                conn, params=(sym,)
            )
            if not r.empty:
                latest_rows[sym] = r.iloc[0]
        except Exception as e:
            logger.error(f"DB read error {sym}: {e}")

    index_syms = [s for s in all_symbols if s in INDEX_DISPLAY]
    stock_syms = [s for s in all_symbols if s not in INDEX_DISPLAY]

    latest_dt = pd.read_sql_query(
        "SELECT MAX(updated_at) AS last_updated FROM technical_indicators", conn
    ).iloc[0, 0] or "—"

    lines = [
        "# 📊 Small Cap Technical Indicators\n\n",
        f"**Last updated:** {latest_dt} IST\n\n",
        "---\n\n",
        "## 📊 MARKET INDEXES\n\n",
        "| Symbol | Datetime | Close | Volume | RSI | EMA20 | MACD | VWAP | Signal |\n",
        "|--------|----------|------:|-------:|----:|------:|-----:|-----:|:------:|\n",
    ]

    for sym in index_syms:
        if sym not in latest_rows:
            continue
        r   = latest_rows[sym]
        sig = r["signal"]
        icon = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}.get(sig, sig)
        vol = "—" if (r["volume"] is None or (isinstance(r["volume"], float) and np.isnan(r["volume"]))) else f"{int(r['volume']):,}"
        lines.append(
            f"| {INDEX_DISPLAY[sym]} | {r['datetime']} "
            f"| {_fmt(r['close'])} | {vol} | {_fmt(r['rsi_14'])} "
            f"| {_fmt(r['ema_20'])} | {_fmt(r['macd'], 4)} | {_fmt(r['vwap'])} | {icon} |\n"
        )

    lines += [
        "\n---\n\n",
        "## 📈 STOCKS\n\n",
        "| Symbol | Datetime | Close | Volume | RSI | EMA20 | MACD | VWAP | Signal |\n",
        "|--------|----------|------:|-------:|----:|------:|-----:|-----:|:------:|\n",
    ]

    for sym in stock_syms:
        if sym not in latest_rows:
            continue
        r    = latest_rows[sym]
        sig  = r["signal"]
        icon = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}.get(sig, sig)
        vol  = "—" if (r["volume"] is None or (isinstance(r["volume"], float) and np.isnan(r["volume"]))) else f"{int(r['volume']):,}"
        lines.append(
            f"| {_dn(sym)} | {r['datetime']} "
            f"| {_fmt(r['close'])} | {vol} | {_fmt(r['rsi_14'])} "
            f"| {_fmt(r['ema_20'])} | {_fmt(r['macd'], 4)} | {_fmt(r['vwap'])} | {icon} |\n"
        )

    lines.append("\n---\n\n")

    for sym in stock_syms:
        if sym not in latest_rows:
            continue
        r    = latest_rows[sym]
        sig  = r["signal"]
        icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")

        lines.append(f"## {_dn(sym)}\n\n")
        lines.append(
            f"**Date:** `{r['datetime']}` &nbsp;|&nbsp; "
            f"**Close:** `{_fmt(r['close'])}` &nbsp;|&nbsp; "
            f"**Signal:** {icon} **{sig}**\n\n"
        )

        lines.append("### 📈 Trend Indicators\n\n")
        _table3(lines, [
            ("SMA 5",         _fmt(r["sma_5"])),
            ("SMA 10",        _fmt(r["sma_10"])),
            ("SMA 20",        _fmt(r["sma_20"])),
            ("SMA 50",        _fmt(r["sma_50"])),
            ("SMA 100",       _fmt(r["sma_100"])),
            ("SMA 200",       _fmt(r["sma_200"])),
            ("EMA 5",         _fmt(r["ema_5"])),
            ("EMA 10",        _fmt(r["ema_10"])),
            ("EMA 20",        _fmt(r["ema_20"])),
            ("EMA 50",        _fmt(r["ema_50"])),
            ("EMA 100",       _fmt(r["ema_100"])),
            ("EMA 200",       _fmt(r["ema_200"])),
            ("WMA 10",        _fmt(r["wma_10"])),
            ("WMA 20",        _fmt(r["wma_20"])),
            ("MACD",          _fmt(r["macd"], 4)),
            ("MACD Signal",   _fmt(r["macd_signal"], 4)),
            ("MACD Diff",     _fmt(r["macd_diff"], 4)),
            ("ADX",           _fmt(r["adx"])),
            ("ADX+",          _fmt(r["adx_pos"])),
            ("ADX-",          _fmt(r["adx_neg"])),
            ("Aroon Up",      _fmt(r["aroon_up"])),
            ("Aroon Down",    _fmt(r["aroon_down"])),
            ("Aroon Ind",     _fmt(r["aroon_indicator"])),
            ("CCI",           _fmt(r["cci"])),
            ("DPO",           _fmt(r["dpo"])),
            ("Mass Index",    _fmt(r["mass_index"])),
            ("Ichimoku A",    _fmt(r["ichimoku_a"])),
            ("Ichimoku B",    _fmt(r["ichimoku_b"])),
            ("Ichimoku Base", _fmt(r["ichimoku_base"])),
            ("Ichimoku Conv", _fmt(r["ichimoku_conv"])),
            ("PSAR",          _fmt(r["psar"])),
            ("STC",           _fmt(r["stc"])),
            ("TRIX",          _fmt(r["trix"], 4)),
            ("Vortex +",      _fmt(r["vortex_pos"])),
            ("Vortex -",      _fmt(r["vortex_neg"])),
        ])

        lines.append("\n### 🌡️ Volatility Indicators\n\n")
        _table3(lines, [
            ("KC Upper",    _fmt(r["kc_upper"])),
            ("KC Middle",   _fmt(r["kc_middle"])),
            ("KC Lower",    _fmt(r["kc_lower"])),
            ("DC Upper",    _fmt(r["dc_upper"])),
            ("DC Middle",   _fmt(r["dc_middle"])),
            ("DC Lower",    _fmt(r["dc_lower"])),
            ("ATR",         _fmt(r["atr"])),
            ("BB Upper",    _fmt(r["bb_upper"])),
            ("BB Middle",   _fmt(r["bb_middle"])),
            ("BB Lower",    _fmt(r["bb_lower"])),
            ("BB %B",       _fmt(r["bb_pband"], 4)),
            ("BB Width",    _fmt(r["bb_wband"], 4)),
            ("Ulcer Index", _fmt(r["ulcer_index"])),
        ])

        lines.append("\n### ⚡ Momentum Indicators\n\n")
        _table3(lines, [
            ("RSI 7",       _fmt(r["rsi_7"])),
            ("RSI 14",      _fmt(r["rsi_14"])),
            ("RSI 21",      _fmt(r["rsi_21"])),
            ("Stoch %K",    _fmt(r["stoch_k"])),
            ("Stoch %D",    _fmt(r["stoch_d"])),
            ("ROC",         _fmt(r["roc"], 4)),
            ("Williams %R", _fmt(r["williams_r"])),
            ("Awe. Osc.",   _fmt(r["awesome_oscillator"], 4)),
            ("KAMA",        _fmt(r["kama"])),
            ("PPO",         _fmt(r["ppo"], 4)),
            ("PVO",         _fmt(r["pvo"], 4)),
            ("TSI",         _fmt(r["tsi"], 4)),
            ("Ult. Osc.",   _fmt(r["ultimate_oscillator"])),
        ])

        lines.append("\n### 📦 Volume Indicators\n\n")
        _table3(lines, [
            ("OBV",         _fmt(r["obv"], 0)),
            ("CMF",         _fmt(r["cmf"], 4)),
            ("Acc/Dist",    _fmt(r["acc_dist"], 0)),
            ("MFI",         _fmt(r["mfi"])),
            ("Force Index", _fmt(r["force_index"], 0)),
            ("EOM",         _fmt(r["eom"], 6)),
            ("VPT",         _fmt(r["vpt"], 0)),
            ("NVI",         _fmt(r["nvi"])),
            ("VWAP",        _fmt(r["vwap"])),
        ])

        lines.append("\n### 🕯️ Price Action\n\n")
        lines.append("| Price Chg % |\n")
        lines.append("|------------:|\n")
        lines.append(f"| {_fmt(r['price_change_pct'], 4)} |\n")

        lines.append("\n---\n\n")

    conn.close()
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("README updated.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(SYMBOLS_FILE):
        logger.error(f"{SYMBOLS_FILE} not found")
        return

    with open(SYMBOLS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_symbols = data.get("Small Cap", []) + data.get("Indexes", [])
    init_db()
    logger.info(f"Processing {len(all_symbols)} symbols")

    for symbol in all_symbols:
        logger.info(f"Processing: {symbol}")
        df = fetch(symbol)
        if df is None or df.empty:
            continue
        try:
            ind_df = build_indicator_df(symbol, df)
            upsert_df(ind_df)
            logger.info(f"  {symbol}: {len(ind_df)} rows stored in DB")
        except Exception as e:
            logger.error(f"  Error {symbol}: {e}", exc_info=True)

    update_readme(all_symbols)
    logger.info("All done.")


if __name__ == "__main__":
    main()
