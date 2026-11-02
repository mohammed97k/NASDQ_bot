import os
import json
import time
import requests
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ═══════════ ENV & CONFIG ═══════════
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

# ═══════════ STRATEGY CONSTANTS (مطابقة للبوت الأول) ═══════════
MULT = 1.0
N_MODELS = 5

MIN_SL_PTS = 40.0
MAX_SL_PTS = 150.0
SL_BUF_PTS = 10.0
COOLDOWN = 10                    # 10 bars × 30min = 5 ساعات
MAX_BARS_TRADE = 60              # 60 bars × 30min = 30 ساعة
USE_INV = True
TP1_PTS = 100.0
TP2_PTS = 200.0                  # ✅ مطابق للبوت الأول
TP3_PTS = 300.0                  # ✅ مطابق للبوت الأول
MAX_DIST_FOR_LIMIT_PTS = 40.0    # ✅ فلتر القرب
EXPIRY_MINUTES = 60              # ✅ صلاحية الأمر المعلق

NAMES = ["Mitigation", "SMR", "OB-Pro", "MMBM", "FVG-Direct"]

SYMBOL_MAP = {
    "NASDAQ": ["USTEC_x100m", "USTECm", "USTEC_x100", "USTEC", "NDXUSD#", "NDXUSD"],
    "DOW": ["US30_x100m", "US30m", "US30_x10m", "US30", "DJI"]
}

# ═══════════ HELPERS ═══════════
def fmt_mosul(dt_utc):
    return dt_utc.astimezone(MOSUL_TZ).strftime('%I:%M %p')

def parse_iso(s):
    if not s: return None
    try: return datetime.fromisoformat(s)
    except: return None

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
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                account_id = r.json().get("accountId")
                print("Session opened: " + str(account_id))
                return account_id
            else:
                print("Session failed: " + str(r.status_code))
                if attempt < max_retries: time.sleep(5)
        except Exception as e:
            print("Session error: " + str(e))
            if attempt < max_retries: time.sleep(5)
    return None

def detect_symbol(account_id, target_idx):
    url = BASE_URL + "/v1/accounts/" + account_id + "/symbols"
    headers = {"Authorization": "Bearer " + TICKERALL_API_KEY}
    try:
        r = requests.get(url, headers=headers, timeout=60)
        data = r.json()
        syms = data.get("symbols", data.get("watched", []))
        for candidate in SYMBOL_MAP[target_idx]:
            if candidate in syms: return candidate
        for s in syms:
            su = str(s).upper()
            if target_idx == "NASDAQ" and ("USTEC" in su or "NDX" in su or "NAS" in su): return str(s)
            if target_idx == "DOW" and ("US30" in su or "DJI" in su or "DOW" in su): return str(s)
    except Exception as e:
        print(f"detect_symbol {target_idx} error: " + str(e))
    return None

def fetch_candles(account_id, symbol, timeframe, limit=500):
    url = BASE_URL + "/v1/accounts/" + account_id + "/candles"
    headers = {"Authorization": "Bearer " + TICKERALL_API_KEY}
    params = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=90)
        if r.status_code != 200: return None
        data = r.json()
        candles = data.get("candles", [])
        if not candles: return None
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df[["datetime", "open", "high", "low", "close"]].copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        return None

# ═══════════ STATE ═══════════
def get_fresh_state():
    return {
        "cntTotal": 0, "cntTP1": 0, "cntTP2": 0, "cntTP3": 0, "cntSL": 0, "cntPts": 0.0,
        "aOn": [False]*N_MODELS, "aLong": [False]*N_MODELS, "aE": [None]*N_MODELS,
        "aSL": [None]*N_MODELS, "aAnc": [None]*N_MODELS, "aT1": [None]*N_MODELS,
        "aT2": [None]*N_MODELS, "aT3": [None]*N_MODELS, "aRk": [None]*N_MODELS,
        "aH1": [False]*N_MODELS, "aH2": [False]*N_MODELS,
        "aTr": [0]*N_MODELS, "aWn": [0]*N_MODELS, "aTim": [""]*N_MODELS,
        "aLastT": [None]*N_MODELS,       # cooldown timestamp
        "aEntryT": [None]*N_MODELS,      # entry timestamp
        "aPend": [False]*N_MODELS,       # pending flag
        "aPendT": [None]*N_MODELS,       # pending timestamp
        "_v": 2
    }

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: s = json.load(f)
            if "NASDAQ" not in s or s.get("_v", 0) < 2:
                print("State version mismatch, resetting...")
                return {"NASDAQ": get_fresh_state(), "DOW": get_fresh_state()}
            for key in ["NASDAQ", "DOW"]:
                if key not in s: s[key] = get_fresh_state()
                else:
                    fresh = get_fresh_state()
                    for f in fresh:
                        if f not in s[key]: s[key][f] = fresh[f]
            return s
        except Exception as e:
            print("State load error: " + str(e))
    return {"NASDAQ": get_fresh_state(), "DOW": get_fresh_state()}

def save_state(s):
    with open(STATE_FILE, "w") as f: json.dump(s, f, indent=2)

# ═══════════ TECHNICAL INDICATORS ═══════════
def calc_indicators(df_m30, df_h1):
    high, low, close = df_m30["high"], df_m30["low"], df_m30["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df_m30["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df_h1["ema200"] = df_h1["close"].ewm(span=200, adjust=False).mean()
    df_h1["ema50"] = df_h1["close"].ewm(span=50, adjust=False).mean()
    h1 = df_h1.iloc[-2]
    trendUp = h1["close"] > h1["ema200"]
    trendDown = h1["close"] < h1["ema200"]
    strongBull = trendUp and h1["close"] > h1["ema50"]
    strongBear = trendDown and h1["close"] < h1["ema50"]
    return df_m30, trendUp, trendDown, strongBull, strongBear

def get_pivots(df, left=3, right=3):
    highs = df["high"].values; lows = df["low"].values; n = len(df)
    sh = np.full(n, np.nan); sl = np.full(n, np.nan)
    for i in range(left, n - right):
        if highs[i] == max(highs[i-left:i+right+1]) and highs[i] > max(highs[i-left:i]):
            sh[i+right] = highs[i]
        if lows[i] == min(lows[i-left:i+right+1]) and lows[i] < min(lows[i-left:i]):
            sl[i+right] = lows[i]
    df["sh"] = sh; df["sl"] = sl
    df["lastSH"] = df["sh"].ffill(); df["lastSL"] = df["sl"].ffill()
    return df

def check_sessions(dt_ny):
    hm = dt_ny.hour * 100 + dt_ny.minute
    londonO = 200 <= hm < 300; sbLDN = 300 <= hm < 400
    judas = 930 <= hm < 1000; sbAM = 1000 <= hm < 1100
    y2022 = 1100 <= hm < 1130; lunch = 1150 <= hm < 1210
    sbPM = 1400 <= hm < 1500; moc = 1515 <= hm < 1545
    fomc = 1400 <= hm < 1430; nfp = 830 <= hm < 900
    lunchBl = 1210 <= hm < 1359
    isFri = dt_ny.weekday() == 4
    tgif = isFri and (1400 <= hm < 1500); amKZ = 930 <= hm < 1130
    anyKZ = londonO or sbLDN or judas or sbAM or y2022 or lunch or sbPM or moc or fomc or nfp or tgif
    sessKZ = anyKZ and not lunchBl
    return sessKZ, amKZ

# ═══════════ CORE STRATEGY ═══════════
def check_strategy(df_m30, df_h1, state, now_utc, idx_name):
    df_m30, trendUp, trendDown, strongBull, strongBear = calc_indicators(df_m30, df_h1)
    df_m30 = get_pivots(df_m30, 3, 3)
    i = len(df_m30) - 1
    row = df_m30.iloc[i]
    now_ny = row["datetime"].tz_convert(NY_TZ)
    sessKZ, amKZ = check_sessions(now_ny)
    time_str = fmt_mosul(now_utc)
    close = float(row["close"]); low = float(row["low"]); high = float(row["high"]); open_p = float(row["open"])

    recentLow = df_m30["low"].iloc[i-20:i].min()
    recentHigh = df_m30["high"].iloc[i-20:i].max()

    bullSweepOK = False; bullSweepLow = None
    for j in range(max(0, i-20), i+1):
        r = df_m30.iloc[j]; rl = df_m30["low"].iloc[max(0, j-20):j].min()
        if r["low"] < rl and r["close"] > rl: bullSweepOK = True; bullSweepLow = r["low"]

    bearSweepOK = False; bearSweepHigh = None
    for j in range(max(0, i-20), i+1):
        r = df_m30.iloc[j]; rh = df_m30["high"].iloc[max(0, j-20):j].max()
        if r["high"] > rh and r["close"] < rh: bearSweepOK = True; bearSweepHigh = r["high"]

    bullMSS = False; bearMSS = False
    for j in range(max(0, i-15), i+1):
        r = df_m30.iloc[j]; p = df_m30.iloc[j-1]
        if not pd.isna(r["lastSH"]) and r["close"] > r["lastSH"] and p["close"] <= r["lastSH"]: bullMSS = True
        if not pd.isna(r["lastSL"]) and r["close"] < r["lastSL"] and p["close"] >= r["lastSL"]: bearMSS = True

    bActive = False; bTop = None; bBot = None; bBar = -100
    sActive = False; sTop = None; sBot = None; sBar = -100
    for j in range(max(0, i-15), i+1):
        if df_m30["low"].iloc[j] > df_m30["high"].iloc[j-2]:
            bTop = df_m30["low"].iloc[j]; bBot = df_m30["high"].iloc[j-2]; bBar = j; bActive = True
        if df_m30["high"].iloc[j] < df_m30["low"].iloc[j-2]:
            sTop = df_m30["low"].iloc[j-2]; sBot = df_m30["high"].iloc[j]; sBar = j; sActive = True
    bFVG_Age = i - bBar if bActive else 999
    sFVG_Age = i - sBar if sActive else 999
    bCE = (bTop + bBot)/2 if bActive else None
    sCE = (sTop + sBot)/2 if sActive else None

    bOBActive = False; bOBHigh = None; bOBLow = None; bOBMT = None; bOBBar = -100
    sOBActive = False; sOBHigh = None; sOBLow = None; sOBMT = None; sOBBar = -100
    for j in range(max(0, i-15), i+1):
        r = df_m30.iloc[j]; p = df_m30.iloc[j-1]
        if r["close"] > r["open"] and p["close"] < p["open"]:
            bOBHigh = p["high"]; bOBLow = p["low"]; bOBMT = (p["open"]+p["close"])/2; bOBBar = j-1; bOBActive = True
        if r["close"] < r["open"] and p["close"] > p["open"]:
            sOBHigh = p["high"]; sOBLow = p["low"]; sOBMT = (p["open"]+p["close"])/2; sOBBar = j-1; sOBActive = True

    bMitActive = False; bMitHigh = None; bMitLow = None; bMitBar = -100
    sMitActive = False; sMitHigh = None; sMitLow = None; sMitBar = -100
    for j in range(max(0, i-15), i+1):
        r = df_m30.iloc[j]; p = df_m30.iloc[j-1]
        if r["close"] > r["open"] and p["close"] < p["open"] and r["close"] > df_m30["high"].iloc[j-2]:
            bMitHigh = p["high"]; bMitLow = p["low"]; bMitBar = j-1; bMitActive = True
        if r["close"] < r["open"] and p["close"] > p["open"] and r["close"] < df_m30["low"].iloc[j-2]:
            sMitHigh = p["high"]; sMitLow = p["low"]; sMitBar = j-1; sMitActive = True

    bMitCE = (bMitHigh + bMitLow)/2 if bMitActive else None
    sMitCE = (sMitHigh + sMitLow)/2 if sMitActive else None
    swingEq = (row["lastSH"] + row["lastSL"]) / 2.0 if not pd.isna(row["lastSH"]) and not pd.isna(row["lastSL"]) else None
    atr = row["atr14"]

    # ═══════════ 1. إدارة الصفقات النشطة ═══════════
    for m in range(N_MODELS):
        if not state["aOn"][m]: continue
        isL = state["aLong"][m]; e = state["aE"][m]; slv = state["aSL"][m]; anc = state["aAnc"][m]
        t1 = state["aT1"][m]; t2 = state["aT2"][m]; t3 = state["aT3"][m]
        h1_hit = state["aH1"][m]; h2_hit = state["aH2"][m]
        entry_t = parse_iso(state["aEntryT"][m]); done = False

        if entry_t and (now_utc - entry_t).total_seconds() / 1800 >= MAX_BARS_TRADE:
            pv = (TP2_PTS if h2_hit else TP1_PTS) if h1_hit else ((close - e) * MULT if isL else (e - close) * MULT)
            state["cntPts"] += pv; state["aOn"][m] = False
            send_telegram(f"⏱ Time Exit (60 bars) [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nCurrent: {close:.2f}")
            continue

        if isL:
            if not h1_hit and high >= t1:
                h1_hit = True; state["aH1"][m] = True; state["aSL"][m] = e
                state["cntTP1"] += 1; state["aWn"][m] += 1
                send_telegram(f"🎯 TP1 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP1: {t1:.2f}\nMove SL to BE")
            if h1_hit and not h2_hit and high >= t2:
                h2_hit = True; state["aH2"][m] = True; state["cntTP2"] += 1
                send_telegram(f"🎯🎯 TP2 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP2: {t2:.2f}")
            if h2_hit and high >= t3:
                state["cntTP3"] += 1; state["cntPts"] += TP3_PTS; state["aOn"][m] = False; done = True
                send_telegram(f"🎯🎯🎯 TP3 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP3: {t3:.2f}\nTrade Closed.")
            if not done and USE_INV and not h1_hit and close < anc:
                state["cntSL"] += 1; state["cntPts"] += ((close - e) * MULT); state["aOn"][m] = False; done = True
                send_telegram(f"⚠️ Invalidation Exit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nExit: {close:.2f}")
            if not done and low <= state["aSL"][m]:
                if h1_hit:
                    state["cntPts"] += (TP2_PTS if h2_hit else TP1_PTS)
                    send_telegram(f"⚪ SL Hit (BE) [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}")
                else:
                    state["cntSL"] += 1; state["cntPts"] += ((state["aSL"][m] - e) * MULT)
                    send_telegram(f"🛑 SL Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}")
                state["aOn"][m] = False; done = True
        else:
            if not h1_hit and low <= t1:
                h1_hit = True; state["aH1"][m] = True; state["aSL"][m] = e
                state["cntTP1"] += 1; state["aWn"][m] += 1
                send_telegram(f"🎯 TP1 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP1: {t1:.2f}\nMove SL to BE")
            if h1_hit and not h2_hit and low <= t2:
                h2_hit = True; state["aH2"][m] = True; state["cntTP2"] += 1
                send_telegram(f"🎯🎯 TP2 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP2: {t2:.2f}")
            if h2_hit and low <= t3:
                state["cntTP3"] += 1; state["cntPts"] += TP3_PTS; state["aOn"][m] = False; done = True
                send_telegram(f"🎯🎯🎯 TP3 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP3: {t3:.2f}\nTrade Closed.")
            if not done and USE_INV and not h1_hit and close > anc:
                state["cntSL"] += 1; state["cntPts"] += ((e - close) * MULT); state["aOn"][m] = False; done = True
                send_telegram(f"⚠️ Invalidation Exit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nExit: {close:.2f}")
            if not done and high >= state["aSL"][m]:
                if h1_hit:
                    state["cntPts"] += (TP2_PTS if h2_hit else TP1_PTS)
                    send_telegram(f"⚪ SL Hit (BE) [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}")
                else:
                    state["cntSL"] += 1; state["cntPts"] += ((e - state["aSL"][m]) * MULT)
                    send_telegram(f"🛑 SL Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}")
                state["aOn"][m] = False; done = True

    # ═══════════ 2. إدارة الأوامر المعلقة ═══════════
    for m in range(N_MODELS):
        if not state["aPend"][m]: continue
        pend_t = parse_iso(state["aPendT"][m])
        if pend_t and (now_utc - pend_t).total_seconds() > EXPIRY_MINUTES * 60:
            state["aPend"][m] = False; state["aPendT"][m] = None
            send_telegram(f"❌ Pending Expired [{idx_name}] [{NAMES[m]}]\nEntry was: {state['aE'][m]:.2f}")
            continue
        e = state["aE"][m]; is_long = state["aLong"][m]; filled = False
        if is_long and low <= e: filled = True
        if not is_long and high >= e: filled = True
        if filled:
            state["aPend"][m] = False; state["aPendT"][m] = None; state["aOn"][m] = True
            state["aEntryT"][m] = now_utc.isoformat(); state["aLastT"][m] = now_utc.isoformat()
            dir_str = "🟢 BUY" if is_long else "🔴 SELL"
            send_telegram(f"✅ Filled! {dir_str} [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}\nTP1: {state['aT1'][m]:.2f}\nTP2: {state['aT2'][m]:.2f}\nTP3: {state['aT3'][m]:.2f}")

    # ═══════════ 3. كشف الإشارات الجديدة ═══════════
    bSig = [False]*5; sSig = [False]*5
    bEA = [None]*5; bSA = [None]*5; sEA = [None]*5; sSA = [None]*5

    if sessKZ and strongBull and bMitActive and (i - bMitBar) <= 8 and close > open_p and close >= bMitCE and (close - open_p) > atr*0.4:
        bSig[0] = True; bEA[0] = bMitCE; bSA[0] = bMitLow - atr*0.3
    if sessKZ and strongBear and sMitActive and (i - sMitBar) <= 8 and close < open_p and close <= sMitCE and (open_p - close) > atr*0.4:
        sSig[0] = True; sEA[0] = sMitCE; sSA[0] = sMitHigh + atr*0.3

    if sessKZ and bearSweepOK and bullMSS and bActive and close >= bCE and (close - open_p) > atr*0.5:
        bSig[1] = True; bEA[1] = bCE; bSA[1] = bBot - atr*0.3
    if sessKZ and bullSweepOK and bearMSS and sActive and close <= sCE and (open_p - close) > atr*0.5:
        sSig[1] = True; sEA[1] = sCE; sSA[1] = sTop + atr*0.3

    if sessKZ and strongBull and bOBActive and (i - bOBBar) <= 10 and bullMSS and swingEq and bOBMT < swingEq and close > open_p and close >= bOBMT:
        bSig[2] = True; bEA[2] = bOBMT; bSA[2] = bOBLow - atr*0.3
    if sessKZ and strongBear and sOBActive and (i - sOBBar) <= 10 and bearMSS and swingEq and sOBMT > swingEq and close < open_p and close <= sOBMT:
        sSig[2] = True; sEA[2] = sOBMT; sSA[2] = sOBHigh + atr*0.3

    if amKZ and bullSweepOK and bullMSS and bOBActive and swingEq and bOBMT < swingEq and trendUp and close >= bOBMT:
        bSig[3] = True; bEA[3] = bOBMT; bSA[3] = bOBLow - atr*0.3

    bBodySize = abs(close - open_p)
    if sessKZ and strongBull and bActive and bFVG_Age <= 5 and close >= bCE and close > bBot and bBodySize > atr*0.3:
        bSig[4] = True; bEA[4] = bCE; bSA[4] = bBot - atr*0.3
    if sessKZ and strongBear and sActive and sFVG_Age <= 5 and close <= sCE and close < sTop and bBodySize > atr*0.3:
        sSig[4] = True; sEA[4] = sCE; sSA[4] = sTop + atr*0.3

    for m in range(N_MODELS):
        if state["aOn"][m] or state["aPend"][m]: continue
        last_t = parse_iso(state["aLastT"][m])
        if last_t and (now_utc - last_t).total_seconds() / 1800 < COOLDOWN: continue

        is_buy = False; e0 = None; s0 = None; anc = None; t1 = None; t2 = None; t3 = None
        if bSig[m]:
            is_buy = True; e0 = bEA[m]; s0 = bSA[m] - (SL_BUF_PTS / MULT)
            anc = s0 + (SL_BUF_PTS / MULT) + atr*0.3
            t1 = e0 + (TP1_PTS / MULT); t2 = e0 + (TP2_PTS / MULT); t3 = e0 + (TP3_PTS / MULT)
        elif sSig[m]:
            is_buy = False; e0 = sEA[m]; s0 = sSA[m] + (SL_BUF_PTS / MULT)
            anc = s0 - (SL_BUF_PTS / MULT) - atr*0.3
            t1 = e0 - (TP1_PTS / MULT); t2 = e0 - (TP2_PTS / MULT); t3 = e0 - (TP3_PTS / MULT)
        else: continue

        d = abs(e0 - s0) * MULT
        if not (MIN_SL_PTS <= d <= MAX_SL_PTS): continue
        dist = abs(e0 - close) * MULT
        if dist > MAX_DIST_FOR_LIMIT_PTS: continue

        state["aLong"][m] = is_buy; state["aE"][m] = e0; state["aSL"][m] = s0; state["aAnc"][m] = anc
        state["aT1"][m] = t1; state["aT2"][m] = t2; state["aT3"][m] = t3; state["aRk"][m] = d
        state["aH1"][m] = False; state["aH2"][m] = False
        state["aTim"][m] = time_str; state["aTr"][m] += 1; state["cntTotal"] += 1

        is_market = (close <= e0) if is_buy else (close >= e0)
        if is_market:
            state["aOn"][m] = True; state["aPend"][m] = False; state["aPendT"][m] = None
            state["aEntryT"][m] = now_utc.isoformat(); state["aLastT"][m] = now_utc.isoformat()
            dir_str = "🟢 BUY (Market)" if is_buy else "🔴 SELL (Market)"
            send_telegram(f"{dir_str} [{idx_name}] [{NAMES[m]}]\nEntry: {e0:.2f}\nSL: {s0:.2f}\nTP1: {t1:.2f}\nTP2: {t2:.2f}\nTP3: {t3:.2f}\n🕐 {time_str}")
        else:
            state["aOn"][m] = False; state["aPend"][m] = True; state["aPendT"][m] = now_utc.isoformat()
            dir_str = "🟢 BUY LIMIT" if is_buy else "🔴 SELL LIMIT"
            send_telegram(f"⏳ {dir_str} [{idx_name}] [{NAMES[m]}]\nEntry: {e0:.2f}\nCurrent: {close:.2f}\nSL: {s0:.2f}\nTP1: {t1:.2f}\nTP2: {t2:.2f}\nTP3: {t3:.2f}\n(Expires in 60 min)\n🕐 {time_str}")

    return state

# ═══════════ MAIN ═══════════
def main():
    print("ICT NASDAQ & DOW Elite Bot v2 - start")
    state = load_state()
    now_utc = datetime.now(ZoneInfo("UTC"))
    now_ny = now_utc.astimezone(NY_TZ)
    if now_ny.weekday() >= 5:
        print("Weekend - disabled"); return

    account_id = open_session()
    if not account_id: return

    for idx_name in ["NASDAQ", "DOW"]:
        print(f"--- Checking {idx_name} ---")
        sym = detect_symbol(account_id, idx_name)
        if not sym:
            print(f"Symbol not found for {idx_name}"); continue
        print(f"Using symbol: {sym}")

        df_m30 = fetch_candles(account_id, sym, "M30", 500)
        df_h1 = fetch_candles(account_id, sym, "H1", 500)
        if df_m30 is None or df_h1 is None:
            print(f"Failed to fetch data for {idx_name}"); continue

        state[idx_name] = check_strategy(df_m30, df_h1, state[idx_name], now_utc, idx_name)
        s = state[idx_name]
        print(f"{idx_name} Stats -> Trades: {s['cntTotal']} | TP1: {s['cntTP1']} | SL: {s['cntSL']} | Points: {s['cntPts']:.2f}")

    save_state(state)

if __name__ == "__main__":
    main()