"""
BTC 200 EMA Signal Bot - Cloud (GitHub Actions) version
===========================================================
Same validated strategy as btc_signal_bot.py, but designed to run as a
SINGLE CHECK per execution (not a continuous loop) - GitHub Actions will
call this script every 5 minutes via a cron schedule, so no laptop/phone
needs to stay on.

Secrets (BOT_TOKEN, CHAT_ID, CAPITAL_INR, RISK_PERCENT) are read from
environment variables, which GitHub Actions injects from repo Secrets -
nothing sensitive is hardcoded here, so this file is safe to put in a
PUBLIC repo (needed for unlimited free GitHub Actions minutes).

Requirements:
    pip install pandas requests
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------- CONFIG FROM ENVIRONMENT (set as GitHub Secrets) ---------------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CAPITAL_INR = float(os.environ.get("CAPITAL_INR", "100000"))
RISK_PERCENT = float(os.environ.get("RISK_PERCENT", "2.0"))

# ----------------------------- VALIDATED STRATEGY CONFIG (do not change lightly) ---------------------------------
TREND_PERSISTENCE = 60
MAX_BREAK_CANDLES = 7
EXTENSION_ATR_MULT = 1.5
RISK_REWARD = 3.0
BUY_ONLY = True

EMA_PERIOD = 200
ATR_PERIOD = 14
MAX_WAIT_FOR_RETEST = 288
SL_BUFFER_PCT = 0.0005
MIN_SL_PERCENT = 0.001
CONTRACT_SIZE_BTC = 1.0     # Exness BTCUSD standard: 1 lot = 1 BTC
MIN_LOT = 0.01
LOT_STEP = 0.01
FALLBACK_USDINR = 88.0

SYMBOL = "BTCUSDT"
LOOKBACK_DAYS = 15
FRESH_SIGNAL_WINDOW_MINUTES = 20   # tolerate some scheduling delay from GitHub Actions
ALERTED_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerted_signals.json")

BINANCE_BASE = "https://api.binance.com/api/v3/klines"
TELEGRAM_BASE = "https://api.telegram.org/bot{token}/sendMessage"


# ----------------------------- ALERTED-SIGNAL TRACKING ---------------------------------
def load_alerted():
    if os.path.exists(ALERTED_LOG_FILE):
        try:
            with open(ALERTED_LOG_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_alerted(alerted_set):
    # keep the file small - only retain the last 500 entries
    trimmed = sorted(alerted_set)[-500:]
    with open(ALERTED_LOG_FILE, "w") as f:
        json.dump(trimmed, f)


# ----------------------------- DATA FETCH ---------------------------------
def fetch_klines(symbol, interval, start_ms, end_ms):
    all_rows = []
    cur = start_ms
    limit = 1000
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cur,
                  "endTime": end_ms, "limit": limit}
        resp = requests.get(BINANCE_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        cur = data[-1][0] + 1
        if len(data) < limit:
            break
        time.sleep(0.2)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "qav", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def get_recent_data():
    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)
    return fetch_klines(SYMBOL, "5m", int(start.timestamp() * 1000), int(end.timestamp() * 1000))


# ----------------------------- INDICATORS ---------------------------------
def add_indicators(df):
    df = df.copy()
    df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    return df


# ----------------------------- STRATEGY ---------------------------------
def detect_signals(m5):
    signals = []
    warmup = EMA_PERIOD + ATR_PERIOD + TREND_PERSISTENCE + 5
    if len(m5) <= warmup:
        return pd.DataFrame(signals)

    state = "SEEK"
    trend_side = None
    side_count = 0
    break_idx = None
    original_trend_side = None

    n = len(m5)
    ema_arr = m5["ema"].values
    atr_arr = m5["atr"].values
    high_arr = m5["high"].values
    low_arr = m5["low"].values
    open_arr = m5["open"].values
    close_arr = m5["close"].values
    time_arr = m5["open_time"].values

    i = warmup
    while i < n:
        side = "above" if close_arr[i] > ema_arr[i] else "below"

        if state == "SEEK":
            if side == trend_side:
                side_count += 1
            else:
                trend_side = side
                side_count = 1
            if side_count >= TREND_PERSISTENCE:
                state = "TRENDING"

        elif state == "TRENDING":
            if side != trend_side:
                state = "BROKEN"
                break_idx = i
                original_trend_side = trend_side

        elif state == "BROKEN":
            dist = abs(close_arr[i] - ema_arr[i])
            if dist >= EXTENSION_ATR_MULT * atr_arr[i]:
                state = "EXTENDED"
            elif i - break_idx > MAX_BREAK_CANDLES:
                state = "SEEK"
                trend_side = side
                side_count = 1

        elif state == "EXTENDED":
            retest_ok, direction = False, None
            if original_trend_side == "above":
                if high_arr[i] > ema_arr[i] and max(open_arr[i], close_arr[i]) < ema_arr[i]:
                    retest_ok, direction = True, "sell"
            else:
                if low_arr[i] < ema_arr[i] and min(open_arr[i], close_arr[i]) > ema_arr[i]:
                    retest_ok, direction = True, "buy"

            if retest_ok:
                if not (BUY_ONLY and direction != "buy"):
                    entry_price = close_arr[i]
                    sl_price = high_arr[i] * (1 + SL_BUFFER_PCT) if direction == "sell" \
                        else low_arr[i] * (1 - SL_BUFFER_PCT)
                    risk = abs(entry_price - sl_price)
                    min_risk = entry_price * MIN_SL_PERCENT
                    if risk < min_risk:
                        sl_price = entry_price - min_risk if direction == "buy" else entry_price + min_risk
                        risk = min_risk
                    target_price = entry_price + RISK_REWARD * risk if direction == "buy" \
                        else entry_price - RISK_REWARD * risk

                    signals.append({
                        "entry_time": pd.Timestamp(time_arr[i]), "direction": direction,
                        "entry": round(entry_price, 2), "sl": round(sl_price, 2),
                        "target": round(target_price, 2)
                    })
                state = "SEEK"
                trend_side = side
                side_count = 1
            elif i - break_idx > MAX_WAIT_FOR_RETEST:
                state = "SEEK"
                trend_side = side
                side_count = 1

        i += 1

    return pd.DataFrame(signals)


# ----------------------------- POSITION SIZING ---------------------------------
def get_usd_inr_rate():
    try:
        resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        resp.raise_for_status()
        return float(resp.json()["rates"]["INR"])
    except Exception as e:
        print(f"USD/INR fetch failed ({e}), using fallback rate {FALLBACK_USDINR}")
        return FALLBACK_USDINR


def calculate_lot_size(entry, sl):
    usdinr = get_usd_inr_rate()
    risk_inr = CAPITAL_INR * (RISK_PERCENT / 100.0)
    risk_usd = risk_inr / usdinr
    sl_distance_usd = abs(entry - sl)
    if sl_distance_usd == 0:
        return None, usdinr, risk_inr
    raw_lots = risk_usd / (sl_distance_usd * CONTRACT_SIZE_BTC)
    lots = round(round(raw_lots / LOT_STEP) * LOT_STEP, 2)
    if lots < MIN_LOT:
        lots = None
    return lots, usdinr, risk_inr


# ----------------------------- TELEGRAM ---------------------------------
def send_telegram_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("BOT_TOKEN / CHAT_ID missing (check GitHub secrets). Message not sent:")
        print(text)
        return
    url = TELEGRAM_BASE.format(token=BOT_TOKEN)
    try:
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        if resp.status_code != 200:
            print(f"Telegram send failed: {resp.status_code} {resp.text}")
        else:
            print("Telegram message sent.")
    except Exception as e:
        print(f"Telegram send error: {e}")


def format_alert(row):
    lots, usdinr, risk_inr = calculate_lot_size(row["entry"], row["sl"])
    if lots is None:
        lot_line = (f"Position size: could not compute (SL too tight for min "
                    f"{MIN_LOT} lot at {RISK_PERCENT}% risk)")
    else:
        lot_line = f"Position size: {lots} lot (~Rs.{risk_inr:.0f} risk @ {RISK_PERCENT}%, USD/INR~{usdinr:.2f})"

    return (
        f"BTC BUY SIGNAL (200 EMA strategy)\n"
        f"Time: {row['entry_time']}\n"
        f"Entry: {row['entry']}\n"
        f"SL: {row['sl']}\n"
        f"Target (1:3): {row['target']}\n"
        f"{lot_line}\n"
        f"(Binance price - verify against Exness before entering. "
        f"Capital assumed: Rs.{CAPITAL_INR:,.0f})"
    )


# ----------------------------- SINGLE-RUN MAIN ---------------------------------
def main():
    print(f"Run started at {datetime.utcnow()}")
    alerted = load_alerted()

    m5_raw = get_recent_data()
    m5 = add_indicators(m5_raw)
    signals = detect_signals(m5)

    now = datetime.utcnow()
    cutoff = pd.Timestamp(now - timedelta(minutes=FRESH_SIGNAL_WINDOW_MINUTES))
    new_count = 0

    for _, row in signals.iterrows():
        key = str(row["entry_time"])
        if row["entry_time"] >= cutoff and key not in alerted:
            send_telegram_message(format_alert(row))
            alerted.add(key)
            new_count += 1

    if new_count:
        save_alerted(alerted)
        print(f"{new_count} new signal(s) alerted.")
    else:
        print("No new signal this run.")


if __name__ == "__main__":
    main()
