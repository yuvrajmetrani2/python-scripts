#!/usr/bin/env python3
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------- Settings ----------
# EDIT THESE TO SUIT YOUR NEEDS
SYMBOLS = ["DOGEUSD", "ETHUSD", "SOLUSD", "BTCUSD"]
#SYMBOL =  "DOGEUSD" # "ETHUSD" "SOLUSD" "BTCUSD" "DOGEUSD"    # confirm exact Delta product symbol
INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d"]
DAYS = 30                                 # historical days to fetch for backtest & smoothing
LOOKBACK_CANDLES = 6                       # consider signals in the last N closed candles
LOOKAHEAD_BARS = 50                        # lookahead window (bars) when evaluating historical TP/SL for probability
MAX_CANDLES_PER_REQUEST = 1000
SLEEP_BETWEEN_CHUNKS = 0.12
TIMEZONE = "Asia/Kolkata"                  # all times converted to IST

# ---------- Utilities ----------
SECS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}

def safe_request(url, params, timeout=20):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# ---------- Fetch OHLC (chunk-safe) ----------
def fetch_ohlc(symbol="DOGEUSD", resolution="1h", limit=500):
    """
    Fetch latest OHLC candles safely (IST timezone).
    limit: number of candles to fetch (default: 500)
    """
    if resolution not in SECS:
        raise ValueError("Unsupported resolution.")

    url = "https://api.india.delta.exchange/v2/history/candles"
    end = int(time.time())
    start = end - limit * SECS[resolution]

    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": start,
        "end": end
    }

    payload = safe_request(url, params)
    candles = payload.get("result", [])

    if not candles:
        print("⚠️ No candles returned.")
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df = df.set_index("time").assign(time=lambda x: x.index)

    return df[["time","open","high","low","close","volume"]]

# ---------- Indicators ----------
def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMA200
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()
    # short SMAs for extra context (optional)
    df["SMA20"] = df["close"].rolling(20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    # MACD (12,26,9)
    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    # Stochastic RSI (0..100)
    rsi14 = RSI(df["close"], 14)
    rsi_min = rsi14.rolling(14).min()
    rsi_max = rsi14.rolling(14).max()
    denom = (rsi_max - rsi_min).replace(0, np.nan)
    stochrsi = 100.0 * (rsi14 - rsi_min) / denom
    stoch_k = stochrsi.rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()
    df["StochRSI"] = stochrsi
    df["StochRSI_k"] = stoch_k
    df["StochRSI_d"] = stoch_d
    return df

# ---------- Generate signals on any provided DF (history or window) ----------
def generate_signals(df: pd.DataFrame):
    """
    Return list of signals found inside df (each signal dict includes tz-aware 'time')
    df may be a subset window (e.g., last N candles) or whole history.
    """
    signals = []
    if df is None or len(df) < 2:
        return signals

    for i in range(1, len(df)):
        time_i = df["time"].iat[i]
        price = df["close"].iat[i]
        ema200 = df["EMA200"].iat[i]
        macd = df["MACD"].iat[i]
        macd_sig = df["MACD_signal"].iat[i]
        stoch_k = df["StochRSI_k"].iat[i]
        stoch_d = df["StochRSI_d"].iat[i]

        # skip if any indicator is NA
        if np.any(pd.isna([price, ema200, macd, macd_sig, stoch_k, stoch_d])):
            continue

        # Bullish: above EMA200, MACD > signal, Stoch K > D but not extreme (>80)
        if (price > ema200) and (macd > macd_sig) and (stoch_k > stoch_d) and (stoch_k < 80):
            prev_low = df["low"].iat[i-1]
            risk = price - prev_low
            if risk <= 0:
                continue
            sl = float(prev_low)
            tp = float(price + 2 * risk)   # 2:1
            rr = (tp - price) / risk
            signals.append({"time": time_i, "side": "LONG", "entry": float(price),
                            "sl": sl, "tp": tp, "rr": round(rr, 3)})

        # Bearish: below EMA200, MACD < signal, Stoch K < D but not extreme (<20)
        elif (price < ema200) and (macd < macd_sig) and (stoch_k < stoch_d) and (stoch_k > 20):
            prev_high = df["high"].iat[i-1]
            risk = prev_high - price
            if risk <= 0:
                continue
            sl = float(prev_high)
            tp = float(price - 2 * risk)
            rr = (price - tp) / risk
            signals.append({"time": time_i, "side": "SHORT", "entry": float(price),
                            "sl": sl, "tp": tp, "rr": round(rr, 3)})
    return signals

# ---------- Backtest signals with limited lookahead (to compute probability) ----------
def backtest_signals_limited(df: pd.DataFrame, signals, lookahead_bars=50):
    """
    For each signal in 'signals' (produced from df), check the next 'lookahead_bars' candles
    to determine whether TP or SL was hit first. Returns a list with 'outcome' added: TP/SL/OPEN.
    """
    results = []
    if not signals:
        return results

    # Map times to row positions for quick lookup
    time_to_idx = {t: i for i, t in enumerate(df["time"].tolist())}

    for sig in signals:
        sig_time = sig["time"]
        if sig_time not in time_to_idx:
            # skip signals whose time isn't found (shouldn't happen)
            continue
        entry_idx = time_to_idx[sig_time]
        hit_tp = False
        hit_sl = False

        # search window
        search_end = min(len(df)-1, entry_idx + lookahead_bars)
        for j in range(entry_idx + 1, search_end + 1):
            high = df["high"].iat[j]
            low = df["low"].iat[j]

            if sig["side"] == "LONG":
                if high >= sig["tp"]:
                    hit_tp = True
                    break
                if low <= sig["sl"]:
                    hit_sl = True
                    break
            else:  # SHORT
                if low <= sig["tp"]:
                    hit_tp = True
                    break
                if high >= sig["sl"]:
                    hit_sl = True
                    break

        outcome = "TP" if hit_tp else ("SL" if hit_sl else "OPEN")
        r = sig.copy()
        r["outcome"] = outcome
        results.append(r)
    return results

# ---------- Get latest signals in the last N candles (IST) and attach probability ----------
def latest_signals_with_probability(df_full: pd.DataFrame, lookback_candles=6, lookahead_bars=50):
    """
    df_full: dataframe with indicators already applied, tz-aware times (IST)
    returns list of latest signals (within last 'lookback_candles' closed candles),
    with a 'probability_%' computed from historical backtest (limited lookahead).
    """
    if df_full is None or len(df_full) < (lookback_candles + 2):
        return []

    # Ensure last candle is closed - drop it if you prefer to ignore the building candle
    # We'll assume API gives closed candles; to be safe drop the last row (in case it's incomplete)
    df = df_full.copy().iloc[:-1]  # drop most recent (possible incomplete) candle

    # compute all historical signals on df (for probability)
    hist_signals = generate_signals(df)
    hist_results = backtest_signals_limited(df, hist_signals, lookahead_bars=lookahead_bars)

    # compute win-rate from closed outcomes
    closed = [r for r in hist_results if r["outcome"] in ("TP", "SL")]
    wins = sum(1 for r in closed if r["outcome"] == "TP")
    losses = sum(1 for r in closed if r["outcome"] == "SL")
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    # window to search for fresh signals: take last (lookback_candles + 1) rows to ensure prev-candle exists
    window = df.iloc[-(lookback_candles + 1):].copy()
    fresh_signals = generate_signals(window)

    # only keep signals whose time falls inside the last lookback_candles rows (exclude older ones)
    # map window times to set for quick check
    window_times = set(window["time"].tolist())
    fresh_signals = [s for s in fresh_signals if s["time"] in window_times]

    # attach probability to each fresh signal
    for s in fresh_signals:
        s["probability_%"] = round(win_rate, 2)

    return fresh_signals, hist_results, win_rate

# ---------- Multi-timeframe runner ----------
def run_all(symbol, intervals=INTERVALS, days=DAYS,
            lookback_candles=LOOKBACK_CANDLES, lookahead_bars=LOOKAHEAD_BARS):
    all_latest = []
    summary = []

    for interval in intervals:
        #print(f"\n--- Processing {symbol} | {interval} ---")
        try:
            df = fetch_ohlc(symbol=symbol, resolution=interval)
        except Exception as e:
            print(f"Fetch error for {interval}: {e}")
            continue

        if df.empty or len(df) < 30:
            print(f"Not enough data for {interval} (rows={len(df)}). Skipping.")
            summary.append({"Interval": interval, "Signals": 0, "Win Rate %": 0.0})
            continue

        df_ind = add_indicators(df)

        # latest signals in last lookback_candles
        fresh_signals, hist_results, win_rate = latest_signals_with_probability(
            df_ind, lookback_candles=lookback_candles, lookahead_bars=lookahead_bars
        )

        # summarize
        total_hist_signals = len(hist_results)
        wins = sum(1 for r in hist_results if r.get("outcome") == "TP")
        losses = sum(1 for r in hist_results if r.get("outcome") == "SL")
        summary.append({"Interval": interval, "Signals": total_hist_signals, "Wins": wins,
                        "Losses": losses, "Win Rate %": round(win_rate, 2)})

        if fresh_signals:
            for s in fresh_signals:
                # convert time to string in IST for printing
                s_out = {
                    "interval": interval,
                    "time_ist": s["time"].strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "side": s["side"],
                    "entry": round(s["entry"], 8),
                    "sl": round(s["sl"], 8),
                    "tp": round(s["tp"], 8),
                    "rr": s["rr"],
                    "prob_%": s["probability_%"]
                }
                all_latest.append(s_out)
                # print("Fresh signal:", s_out)
        else:
            pass
            #print("No fresh signals in last", lookback_candles, "closed candles for", interval)

    summary_df = pd.DataFrame(summary)
    #print("\n=== Summary (per timeframe) ===")
    #print(summary_df.to_string(index=False))

    if all_latest:
        print(f"\n=== All fresh signals (summarized for {symbol}) ===")
        print(pd.DataFrame(all_latest).to_string(index=False))
    else:
        print("\nNo fresh signals across selected timeframes.")

    return summary_df, all_latest

# ---------- Run when executed ----------
if __name__ == "__main__":
    for sym in SYMBOLS:
        run_all(sym)
