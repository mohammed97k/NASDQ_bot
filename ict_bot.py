import os
import json
import time
import requests
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

TICKERALL_API_KEY = os.environ.get("TICKERALL_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_GROUP_CHAT_ID = os.environ.get("TELEGRAM_GROUP_CHAT_ID")
MT5_PASSWORD = os.environ.get("MT5_PASSWORD")
MT5_SERVER = os.environ.get("MT5_SERVER")
MT5_ACCOUNT = os.environ.get("MT5_ACCOUNT")

STATE_FILE = "state.json"
NY_TZ = ZoneInfo("America/New_York")
MOSUL_TZ = ZoneInfo("Asia/Baghdad")
BASE_URL = "https://api.tickerall.com"
SERVER_OFFSET_HOURS = 0
ALERT_COOLDOWN_MINUTES = 60

MULT = 1.0
ATR_PERIOD = 14
MIN_SL_PTS = 50.0
MAX_SL_PTS = 180.0
COOLDOWN_NORMAL = 20
COOLDOWN_STRONG = 5
MAX_TRADES_PER_DAY = 3
MAX_BARS_TRADE = 60
TP1_R = 1.0
TP2_R = 2.0
TP3_R = 3.0

SYMBOL_CANDIDATES = ["USTEC_x100m", "USTECm", "USTEC_x100", "USTEC", "NDXUSD#", "NDXUSD"]
SYMBOL = None

PIVOT_LEFT = 4
PIVOT_RIGHT = 4
SWEEP_LB = 20
FVG_EXPIRY = 15
OB_EXPIRY = 15
MSS_EXPIRY = 25
SWEEP_EXPIRY = 25
STALE_DATA_MINUTES = 360


def fmt_mosul(dt_utc):
    return dt_utc.astimezone(MOSUL_TZ).strftime('%I:%M %p')


def send_telegram(message):
    chat_ids = [c for c in [TELEGRAM_CHAT_ID, TELEGRAM_GROUP_CHAT_ID] if c]
    for chat_id in chat_ids:
        url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
            print("TG->" + str(chat_id) + ": ok=" + str(r.json().get('ok')))
        except Exception as e:
            print("TG Error: " + str(e))


def open_session(max_retries=3):
    url = BASE_URL + "/v1/sessions"
    headers = {"Authorization": "Bearer " + TICKERALL_API_KEY, "Content-Type": "application/json"}
    payload = {
        "broker": "mt5",
        "server": MT5_SERVER,
        "account": int(MT5_ACCOUNT) if MT5_ACCOUNT and MT5_ACCOUNT.isdigit() else MT5_ACCOUNT,
        "password": MT5_PASSWORD
    }
    for attempt in range(1, max_retries + 1):
        try:
            print("Session attempt " + str(attempt) + "/" + str(max_retries))
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                account_id = r.json().get("accountId")
                print("Session opened: " + str(account_id))
                return account_id
            else:
                print("Session failed: " + str(r.status_code) + " " + r.text[:200])
                if attempt < max_retries:
                    time.sleep(5)
        except Exception as e:
            print("Session error: " + str(e))
            if attempt < max_retries:
                time.sleep(5)
    print("All session attempts failed")
    return None


def detect_symbol(account_id):
    url = BASE_URL + "/v1/accounts/" + account_id + "/symbols"
    headers = {"Authorization": "Bearer " + TICKERALL_API_KEY}
    try:
        r = requests.get(url, headers=headers, timeout=60)
        print("Symbols response: " + r.text[:500])
        data = r.json()
        syms = data.get("symbols", data.get("watched", []))
        print("Available: " + str(syms))
        for candidate in SYMBOL_CANDIDATES:
            if candidate in syms:
                print("Matched: " + candidate)
                return candidate
        for s in syms:
            su = str(s).upper()
            if "USTEC" in su or "NDX" in su or "NAS" in su:
                print("Fallback matched: " + str(s))
                return str(s)
    except Exception as e:
        print("detect_symbol error: " + str(e))
    return None


def fetch_candles(account_id, symbol, timeframe, limit=500, hours=500):
    url = BASE_URL + "/v1/accounts/" + account_id + "/candles"
    headers = {"Authorization": "Bearer " + TICKERALL_API_KEY}
    params = {"symbol": symbol, "timeframe": timeframe, "limit": limit, "hours": hours}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=90)
        if r.status_code != 200:
            print("Fail " + timeframe + ": " + str(r.status_code) + " " + r.text[:200])
            return None
        data = r.json()
        candles = data.get("candles", [])
        if not candles:
            print("No candles for " + timeframe + " symbol=" + symbol)
            return None
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True) - pd.Timedelta(hours=SERVER_OFFSET_HOURS)
        df = df[["datetime", "open", "high", "low", "close"]].copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c])
        df = df.sort_values("datetime").reset_index(drop=True)
        print("OK " + timeframe + ": " + str(len(df)) + " candles | last: " + str(df.iloc[-1]['close']))
        return df
    except Exception as e:
        print("Fetch " + timeframe + " Error: " + str(e))
        return None


def ta_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def ta_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def ta_pivothigh(highs, left, right):
    n = len(highs)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        window = highs[i - left:i + right + 1]
        if highs[i] == window.max() and (window == highs[i]).sum() == 1:
            result[i + right] = highs[i]
    return result


def ta_pivotlow(lows, left, right):
    n = len(lows)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        window = lows[i - left:i + right + 1]
        if lows[i] == window.min() and (window == lows[i]).sum() == 1:
            result[i + right] = lows[i]
    return result


def ta_lowest(series, length):
    return series.rolling(length).min()


def ta_highest(series, length):
    return series.rolling(length).max()


def tm(dt, h1, m1, h2, m2):
    t = dt.hour * 60 + dt.minute
    return (h1 * 60 + m1) <= t < (h2 * 60 + m2)


def session_flags(dt):
    return {
        "Judas": tm(dt, 9, 30, 9, 45),
        "AM": tm(dt, 9, 30, 11, 30),
        "Lunch": tm(dt, 12, 0, 13, 30),
        "PM": tm(dt, 13, 30, 15, 0),
        "TGIF": (dt.weekday() == 4) and tm(dt, 13, 30, 15, 0),
    }


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            s = json.load(f)
            print("State loaded: active=" + ("YES" if s.get('active_trade') else "NO"))
            return s
    return {"active_trade": None, "last_entry_time": None, "trade_count_today": 0, "last_day": None, "models_done_today": []}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)
    print("State saved")


def manage_trade(state, df5, now_utc):
    t = state["active_trade"]
    if t is None:
        return
    d = t["direction"]
    e = t["entry"]
    tp1 = t["tp1"]
    tp2 = t["tp2"]
    tp3 = t["tp3"]
    ets = t.get("entry_time")
    if not ets:
        return
    entry_dt = datetime.fromisoformat(ets)
    entry_dt_adj = entry_dt - pd.Timedelta(hours=SERVER_OFFSET_HOURS)
    elapsed = (now_utc - entry_dt).total_seconds()
    bars_elapsed = elapsed / 300
    if bars_elapsed >= MAX_BARS_TRADE:
        state["active_trade"] = None
        send_telegram("Time Exit (60 bars)\nModel: " + str(t['model']) + "\nDir: " + d + "\nEntry: " + str(e))
        return
    bars_since = df5[df5["datetime"] >= entry_dt_adj].copy()
    if len(bars_since) == 0:
        return
    for idx, row in bars_since.iterrows():
        bh = float(row["high"])
        bl = float(row["low"])
        bt = row["datetime"]
        csl = t["sl"]
        sl_hit = (d == "BUY" and bl <= csl) or (d == "SELL" and bh >= csl)
        if sl_hit:
            if t["tp1_hit"]:
                send_telegram("SL after TP1 (BE)\nModel: " + str(t['model']) + "\nEntry: " + str(e) + "\nSL: " + str(csl))
            else:
                send_telegram("SL hit!\nModel: " + str(t['model']) + "\nEntry: " + str(e) + "\nSL: " + str(csl))
            state["active_trade"] = None
            return
        if not t["tp1_hit"]:
            hit = (d == "BUY" and bh >= tp1) or (d == "SELL" and bl <= tp1)
            if hit:
                t["tp1_hit"] = True
                t["sl"] = e
                send_telegram("TP1 hit!\nModel: " + str(t['model']) + "\nEntry: " + str(e) + "\nTP1: " + str(tp1) + "\nMove SL to BE: " + str(e))
        if t["tp1_hit"] and not t["tp2_hit"]:
            hit = (d == "BUY" and bh >= tp2) or (d == "SELL" and bl <= tp2)
            if hit:
                t["tp2_hit"] = True
                send_telegram("TP2 hit!\nModel: " + str(t['model']) + "\nEntry: " + str(e) + "\nTP2: " + str(tp2))
        if t["tp2_hit"] and not t["tp3_hit"]:
            hit = (d == "BUY" and bh >= tp3) or (d == "SELL" and bl <= tp3)
            if hit:
                t["tp3_hit"] = True
                send_telegram("TP3 hit!\nModel: " + str(t['model']) + "\nEntry: " + str(e) + "\nTP3: " + str(tp3))
                state["active_trade"] = None
                return


def build_context(df):
    n = len(df)
    atr = ta_atr(df, ATR_PERIOD).values
    sh = ta_pivothigh(df["high"].values, PIVOT_LEFT, PIVOT_RIGHT)
    sl = ta_pivotlow(df["low"].values, PIVOT_LEFT, PIVOT_RIGHT)

    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    cur_sh = np.nan
    cur_sl = np.nan
    for i in range(n):
        if not np.isnan(sh[i]):
            cur_sh = sh[i]
        if not np.isnan(sl[i]):
            cur_sl = sl[i]
        last_sh[i] = cur_sh
        last_sl[i] = cur_sl

    low_a = df["low"].values
    high_a = df["high"].values
    close_a = df["close"].values
    open_a = df["open"].values

    recent_low = ta_lowest(df["low"], SWEEP_LB).shift(1).values
    recent_high = ta_highest(df["high"], SWEEP_LB).shift(1).values

    bull_sweep = (low_a < recent_low) & (close_a > recent_low)
    bear_sweep = (high_a > recent_high) & (close_a < recent_high)

    bull_sweep_ok = np.zeros(n, dtype=bool)
    bull_sweep_low = np.full(n, np.nan)
    bear_sweep_ok = np.zeros(n, dtype=bool)
    bear_sweep_high = np.full(n, np.nan)

    cbok = False
    cbbr = -1
    cbbl = np.nan
    cek = False
    cebr = -1
    cebh = np.nan

    for i in range(n):
        if bull_sweep[i]:
            cbok = True
            cbbr = i
            cbbl = low_a[i]
        if bear_sweep[i]:
            cek = True
            cebr = i
            cebh = high_a[i]
        if cbok and (i - cbbr) > SWEEP_EXPIRY:
            cbok = False
        if cek and (i - cebr) > SWEEP_EXPIRY:
            cek = False
        bull_sweep_ok[i] = cbok
        bull_sweep_low[i] = cbbl
        bear_sweep_ok[i] = cek
        bear_sweep_high[i] = cebh

    bull_mss = np.zeros(n, dtype=bool)
    bear_mss = np.zeros(n, dtype=bool)
    cbm = False
    cbmbar = -1
    csm = False
    csmbar = -1

    for i in range(1, n):
        if not np.isnan(last_sh[i]) and close_a[i] > last_sh[i] and close_a[i - 1] <= last_sh[i] and bull_sweep_ok[i]:
            cbm = True
            cbmbar = i
        if not np.isnan(last_sl[i]) and close_a[i] < last_sl[i] and close_a[i - 1] >= last_sl[i] and bear_sweep_ok[i]:
            csm = True
            csmbar = i
        if cbm and (i - cbmbar) > MSS_EXPIRY:
            cbm = False
        if csm and (i - csmbar) > MSS_EXPIRY:
            csm = False
        bull_mss[i] = cbm
        bear_mss[i] = csm

    bOBHigh = np.full(n, np.nan)
    bOBLow = np.full(n, np.nan)
    bOBMT = np.full(n, np.nan)
    bOBActive = np.zeros(n, dtype=bool)
    sOBHigh = np.full(n, np.nan)
    sOBLow = np.full(n, np.nan)
    sOBMT = np.full(n, np.nan)
    sOBActive = np.zeros(n, dtype=bool)

    cboH = np.nan
    cboL = np.nan
    cboMT = np.nan
    cboBar = -1
    cboAct = False
    csoH = np.nan
    csoL = np.nan
    csoMT = np.nan
    csoBar = -1
    csoAct = False

    for i in range(1, n):
        if close_a[i] > open_a[i] and close_a[i - 1] < open_a[i - 1]:
            cboH = high_a[i - 1]
            cboL = low_a[i - 1]
            cboMT = (open_a[i - 1] + close_a[i - 1]) / 2.0
            cboBar = i - 1
            cboAct = True
        if close_a[i] < open_a[i] and close_a[i - 1] > open_a[i - 1]:
            csoH = high_a[i - 1]
            csoL = low_a[i - 1]
            csoMT = (open_a[i - 1] + close_a[i - 1]) / 2.0
            csoBar = i - 1
            csoAct = True
        if cboAct and (i - cboBar) > OB_EXPIRY:
            cboAct = False
        if csoAct and (i - csoBar) > OB_EXPIRY:
            csoAct = False
        bOBHigh[i] = cboH
        bOBLow[i] = cboL
        bOBMT[i] = cboMT
        bOBActive[i] = cboAct
        sOBHigh[i] = csoH
        sOBLow[i] = csoL
        sOBMT[i] = csoMT
        sOBActive[i] = csoAct

    midnightOpen = np.full(n, np.nan)
    df_ny = df["datetime"].dt.tz_convert(NY_TZ)
    cur_mo = np.nan
    for i in range(n):
        dt_ny = df_ny.iloc[i]
        if dt_ny.hour == 0 and dt_ny.minute == 0:
            cur_mo = open_a[i]
        midnightOpen[i] = cur_mo

    return {
        "atr": atr,
        "last_sh": last_sh,
        "last_sl": last_sl,
        "bull_sweep_ok": bull_sweep_ok,
        "bear_sweep_ok": bear_sweep_ok,
        "bull_sweep_low": bull_sweep_low,
        "bear_sweep_high": bear_sweep_high,
        "bull_mss": bull_mss,
        "bear_mss": bear_mss,
        "bOBHigh": bOBHigh,
        "bOBLow": bOBLow,
        "bOBMT": bOBMT,
        "bOBActive": bOBActive,
        "sOBHigh": sOBHigh,
        "sOBLow": sOBLow,
        "sOBMT": sOBMT,
        "sOBActive": sOBActive,
        "midnightOpen": midnightOpen,
    }


def check_signal(df5, df1h, state, now_utc, now_ny):
    try:
        ctx = build_context(df5)
        i = len(df5) - 1
        atr = ctx["atr"][i]
        if np.isnan(atr):
            return None

        df1h = df1h.copy()
        df1h["ema200"] = ta_ema(df1h["close"], 200)
        h1 = df1h.iloc[-2]
        trend_up = h1["close"] > h1["ema200"] if not pd.isna(h1["ema200"]) else False
        trend_down = h1["close"] < h1["ema200"] if not pd.isna(h1["ema200"]) else False
        print("H1 trend: " + ("UP" if trend_up else "DOWN" if trend_down else "NONE"))

        sessions = session_flags(now_ny)
        close = df5["close"].iloc[i]
        low = df5["low"].iloc[i]
        high = df5["high"].iloc[i]
        opn = df5["open"].iloc[i]
        rng = high - low
        strong_up = rng > 0 and (close - low) / rng >= 0.70
        strong_dn = rng > 0 and (high - close) / rng >= 0.70

        midnight = ctx["midnightOpen"][i]
        bbh = ctx["bOBHigh"][i]
        bbl = ctx["bOBLow"][i]
        bsh = ctx["sOBHigh"][i]
        bsl = ctx["sOBLow"][i]

        chainL = ctx["bull_sweep_ok"][i] and ctx["bull_mss"][i]
        chainS = ctx["bear_sweep_ok"][i] and ctx["bear_mss"][i]

        if state["trade_count_today"] >= MAX_TRADES_PER_DAY:
            return None
        if state["last_entry_time"]:
            last_dt = datetime.fromisoformat(state["last_entry_time"])
            bars_since = int((now_utc - last_dt).total_seconds() / 300)
            if bars_since <= COOLDOWN_NORMAL:
                return None

        sigL = False
        sigS = False
        model = None

        if sessions["AM"] and chainL and close > h1["ema200"] and not np.isnan(midnight) and close > midnight and not np.isnan(bbh) and low <= bbh and close > bbl:
            sigL = True
            model = "MMBM"
        elif sessions["AM"] and chainS and close < h1["ema200"] and not np.isnan(midnight) and close < midnight and not np.isnan(bsl) and high >= bsl and close < bsh:
            sigS = True
            model = "MMSM"
        elif sessions["AM"] and not np.isnan(midnight) and low < midnight and close > midnight and ctx["bull_mss"][i]:
            sigL = True
            model = "PO3"
        elif sessions["AM"] and not np.isnan(midnight) and high > midnight and close < midnight and ctx["bear_mss"][i]:
            sigS = True
            model = "PO3"
        elif sessions["Judas"] and ctx["bull_sweep_ok"][i] and close > opn and close > df5["high"].iloc[i - 1] and strong_up and close > h1["ema200"]:
            sigL = True
            model = "Judas"
        elif sessions["Judas"] and ctx["bear_sweep_ok"][i] and close < opn and close < df5["low"].iloc[i - 1] and strong_dn and close < h1["ema200"]:
            sigS = True
            model = "Judas"
        elif sessions["AM"] and chainL and not np.isnan(bbh) and low <= bbh and close > bbl:
            sigL = True
            model = "SMR"
        elif sessions["AM"] and chainS and not np.isnan(bsl) and high >= bsl and close < bsh:
            sigS = True
            model = "SMR"
        elif sessions["TGIF"] and chainL and ctx["bull_mss"][i]:
            sigL = True
            model = "TGIF"
        elif sessions["TGIF"] and chainS and ctx["bear_mss"][i]:
            sigS = True
            model = "TGIF"
        elif sessions["Lunch"] and chainL and not np.isnan(bbh) and close > bbh:
            sigL = True
            model = "Lunch"
        elif sessions["Lunch"] and chainS and not np.isnan(bsl) and close < bsl:
            sigS = True
            model = "Lunch"

        if not sigL and not sigS:
            return None

        if sigL:
            sl_level = bbl - atr * 0.3 if not np.isnan(bbl) else ctx["bull_sweep_low"][i] - atr * 0.3
            entry = close
            direction = "BUY"
        else:
            sl_level = bsh + atr * 0.3 if not np.isnan(bsh) else ctx["bear_sweep_high"][i] + atr * 0.3
            entry = close
            direction = "SELL"

        if np.isnan(entry) or np.isnan(sl_level):
            return None

        sl_pts = abs(entry - sl_level) * MULT
        if sl_pts < MIN_SL_PTS or sl_pts > MAX_SL_PTS:
            return None

        sl_dist = abs(entry - sl_level)
        if direction == "BUY":
            tp1 = entry + sl_dist * TP1_R
            tp2 = entry + sl_dist * TP2_R
            tp3 = entry + sl_dist * TP3_R
        else:
            tp1 = entry - sl_dist * TP1_R
            tp2 = entry - sl_dist * TP2_R
            tp3 = entry - sl_dist * TP3_R

        print(direction + " " + model + " @ " + str(entry))
        return {
            "model": model,
            "direction": direction,
            "entry": round(entry, 2),
            "sl": round(sl_level, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
        }
    except Exception as e:
        print("check_signal Exception: " + str(e))
        traceback.print_exc()
        return None


def main():
    global SYMBOL
    print("NAS100 Bot - start")
    try:
        state = load_state()
        now_utc = datetime.now(ZoneInfo("UTC"))
        now_ny = now_utc.astimezone(NY_TZ)

        if now_ny.weekday() >= 5:
            print("Weekend - disabled")
            save_state(state)
            return

        today = nowmodels_ny.date().isoformat()
        if state["last_day"]_d != today:
            state["trade_count_today"] = 0one
            state["_today"] = []
            state["last_day"] = today

        print("Opening session...")
        account_id = open_session()
        if not account_id:
            save_state(state)
            return

        SYMBOL = detect_symbol(account_id)
        if not SYMBOL:
            print("No matching symbol found")
            save_state(state)
            return
        print("Using symbol: " + SYMBOL)

        if state["active_trade"] is not None:
            print("Managing active trade...")
            df5 = fetch_candles(account_id, SYMBOL, "M5", 500, 42)
            if df5 is None:
                save_state(state)
                return
            manage_trade(state, df5, now_utc)
            save_state(state)
            return

        print("Fetch M5...")
        df5 = fetch_candles(account_id, SYMBOL, "M5", 500, 42)
        print("Fetch H1...")
        df1h = fetch_candles(account_id, SYMBOL, "H1", 500, 500)

        if df5 is None or df1h is None:
            save_state(state)
            return

        last_candle_time = df5.iloc[-1]["datetime"]
        minutes_old = (now_utc - last_candle_time).total_seconds() / 60
        print("Candle age: " + str(round(minutes_old, 1)) + " min")
        if minutes_old > STALE_DATA_MINUTES:
            print("Stale data - skipping")
            save_state(state)
            return

        print("Price: " + str(round(float(df5.iloc[-1]['close']), 2)) + " | Mosul: " + fmt_mosul(now_utc))

        print("Checking signal...")
        sig = check_signal(df5, df1h, state, now_utc, now_ny)
        if sig is None:
            print("No signal")
            save_state(state)
            return

        model = sig["model"]
        if model in state["models_done_today"]:
            print("Model already used today: " + model)
            save_state(state)
            return

        print("Signal found!")
        state["active_trade"] = {
            "model": sig["model"],
            "direction": sig["direction"],
            "entry": sig["entry"],
            "sl": sig["sl"],
            "tp1": sig["tp1"],
            "tp2": sig["tp2"],
            "tp3": sig["tp3"],
            "tp1_hit": False,
            "tp2_hit": False,
            "tp3_hit": False,
            "entry_time": now_utc.isoformat(),
        }
        state["trade_count_today"] += 1
        state["last_entry_time"] = now_utc.isoformat()
        state["models_done_today"].append(model)

        send_telegram(
            "New Signal!\n\n"
            "Model: " + str(sig['model']) + "\n"
            "Direction: " + str(sig['direction']) + "\n"
            "Entry: " + str(sig['entry']) + "\n"
            "Stop: " + str(sig['sl']) + "\n"
            "TP1: " + str(sig['tp1']) + "\n"
            "TP2: " + str(sig['tp2']) + "\n"
            "TP3: " + str(sig['tp3']) + "\n\n"
            "Time: " + fmt_mosul(now_utc) + " (Mosul)"
        )
        save_state(state)
    except Exception as e:
        print("Error: " + str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()