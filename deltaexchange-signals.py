#!/usr/bin/env python3

#NOTE: Install Beep package if OS is linux
#TODO : Get More Frequent Indicators

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import pytz
import platform, os, winsound
import logging



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
log_file_name = "3-trade-signals.log"
# ---------- Utilities ----------
SECS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    
    handlers=[
        logging.FileHandler(log_file_name, mode='w'),  # Write logs to a file
        #logging.StreamHandler()  # Also print logs to the console
    ]
)

'''
def safe_request(url, params, timeout=20):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
'''
    
def safe_request(url, params, retries=3, timeout=20):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"⚠️ Request failed ({i+1}/{retries}): {e}")
            time.sleep(2)
    return {}
    
def beep():
    """
    Generates a simple beep sound.
    
    The function checks the operating system and uses the appropriate
    library to generate the sound.
    """
    system = platform.system()
    
    if system == 'Windows':
        # winsound is a built-in library for Windows.
        #import winsound
        winsound.Beep(2000,200) # Set Frequency in Hertz, Duration in ms
        winsound.Beep(1100,300) # Set Frequency in Hertz, Duration in ms
        #print("Beep sound generated on Windows.")
        
    elif system == 'Linux':
        # On Linux, you can use a command-line tool like 'beep'.
        # This requires the 'beep' package to be installed.
        #import os
        os.system('beep')
        #print("Beep sound generated on Linux.")
        
    elif system == 'Darwin':  # macOS
        # On macOS, you can use the 'say' command to make a sound.
        # This is a bit of a workaround to get a notification sound.
        #import os
        os.system('say " "')
        #print("Beep sound generated on macOS.")
        
    else:
        pass
        #print("Beeper not supported on this operating system.")

# ---------- Fetch OHLC (chunk-safe) ----------
def fetch_ohlc(symbol="DOGEUSD", resolution="5m", limit=500):
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


def analyze_signals(df, resolution):
    signals = []
    for i in range(50, len(df)):
        candle = df.iloc[i]
        time_ist = candle.name.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Indicators
        ema200 = df["ema200"].iloc[i]
        close = candle["close"]
        stoch_k = df["stoch_k"].iloc[i]
        stoch_d = df["stoch_d"].iloc[i]
        macd = df["macd"].iloc[i]
        signal = df["macd_signal"].iloc[i]
        hist = df["macd_hist"].iloc[i]

        # Debug logs (print each condition)
        logging.debug(f"{time_ist} | Close={close:.5f}, EMA200={ema200:.5f}, "
                      f"StochRSI=({stoch_k:.2f},{stoch_d:.2f}), "
                      f"MACD={macd:.6f}, Signal={signal:.6f}, Hist={hist:.6f}")

        # Conditions
        ema_cond = close > ema200
        stoch_cond = (stoch_k > stoch_d) and (stoch_k < 20)   # example bullish
        macd_cond = (macd > signal) and (hist > 0)

        # If signal
        if ema_cond and stoch_cond and macd_cond:
            logging.info(f"LONG Signal at {time_ist} "
                         f"(Triggered by: EMA={ema_cond}, StochRSI={stoch_cond}, MACD={macd_cond})")

            signals.append({
                "time": time_ist,
                "side": "LONG",
                "entry": close,
                "sl": df["low"].iloc[i-1],
                "tp": close * 1.005,
                "prob": 50  # placeholder
            })

    return signals



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


def latest_signals_with_probability(
    df_full: pd.DataFrame, 
    lookback_candles=6, 
    lookahead_bars=50, 
    pip_size=0.0001, 
    lot_size=100000
):
    """
    df_full: dataframe with indicators already applied, tz-aware times (IST)
    returns list of latest signals (within last 'lookback_candles' closed candles),
    with a 'probability_%' computed from historical backtest (limited lookahead).
    Adds pips, risk/reward ratio, and profit per 1 lot.
    """
    if df_full is None or len(df_full) < (lookback_candles + 2):
        return []

    df = df_full.copy().iloc[:-1]  # drop most recent (possible incomplete) candle

    # compute all historical signals on df (for probability)
    hist_signals = generate_signals(df)
    hist_results = backtest_signals_limited(df, hist_signals, lookahead_bars=lookahead_bars)

    # compute win-rate from closed outcomes
    closed = [r for r in hist_results if r["outcome"] in ("TP", "SL")]
    wins = sum(1 for r in closed if r["outcome"] == "TP")
    losses = sum(1 for r in closed if r["outcome"] == "SL")
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    # window to search for fresh signals
    window = df.iloc[-(lookback_candles + 1):].copy()
    fresh_signals = generate_signals(window)

    window_times = set(window["time"].tolist())
    fresh_signals = [s for s in fresh_signals if s["time"] in window_times]

    # pip value for 1 lot
    pip_value_per_lot = pip_size * lot_size

    # attach probability + pip calculations
    for s in fresh_signals:
        s["probability_%"] = round(win_rate, 2)

        entry = s.get("entry", None)
        tp = s.get("tp", None)
        sl = s.get("sl", None)

        if entry and tp:
            s["tp_pips"] = round((tp - entry) / pip_size, 1) if s["side"] == "LONG" else round((entry - tp) / pip_size, 1)
        else:
            s["tp_pips"] = None

        if entry and sl:
            s["sl_pips"] = round((entry - sl) / pip_size, 1) if s["side"] == "LONG" else round((sl - entry) / pip_size, 1)
        else:
            s["sl_pips"] = None

        # Risk/Reward ratio
        if s["tp_pips"] and s["sl_pips"] and s["sl_pips"] != 0:
            s["rr_ratio"] = round(s["tp_pips"] / s["sl_pips"], 2)
        else:
            s["rr_ratio"] = None

        # Profit for 1 lot at TP
        if s["tp_pips"]:
            s["profit_1lot"] = round(s["tp_pips"] * pip_value_per_lot, 2)
        else:
            s["profit_1lot"] = None

        logging.info(
            f"Signal: {s['side']} | Time={s['time']} | Entry={entry}, SL={sl}, TP={tp} | "
            f"TP_pips={s['tp_pips']}, SL_pips={s['sl_pips']}, RR={s['rr_ratio']} | "
            f"Prob={s['probability_%']}% | Profit(1lot)={s['profit_1lot']}"
        )

    return fresh_signals, hist_results, win_rate


'''
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
        logging.info(f"Generated Signal: {s["side"]} at {s["time"]} with probability {s["probability_%"]}")

    return fresh_signals, hist_results, win_rate

'''

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
                enter = " -> " if s["probability_%"] >= 40 else "    "
                #Beep at Strong Signals - win rate of > 40%
                if s["probability_%"] >= 40:
                    beep()
                s_out = {
                    "   ": enter,
                    "int": interval,
                    "time_ist": s["time"].strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "side": s["side"],
                    "entry": round(s["entry"], 8),
                    "sl": round(s["sl"], 8),
                    "tp": round(s["tp"], 8),
                    #"rr": s["rr"],
                    "tp_pips": s["tp_pips"],
                    "sl_pips": s["sl_pips"],
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
    while True:
        for sym in SYMBOLS:
            run_all(sym)
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        
        print(f"\n ----------------- ", now.strftime("%Y-%m-%d %H:%M:%S") , "------------------------------- ")
        
        time.sleep(300)
