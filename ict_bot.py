import os
import json
import time
import requests
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
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

# ═══════════ STRATEGY CONSTANTS ═══════════
MULT = 1.0
N_MODELS = 5

MIN_SL_PTS = 40.0
MAX_SL_PTS = 150.0
SL_BUF_PTS = 10.0
COOLDOWN = 10
MAX_BARS_TRADE = 60
USE_INV = True
TP1_PTS = 100.0
TP2_PTS = 150.0
TP3_PTS = 200.0

NAMES = ["Mitigation", "SMR", "OB-Pro", "MMBM", "FVG-Direct"]

# ═══════════ SYMBOL MAPS ═══════════
SYMBOL_MAP = {
    "NASDAQ": ["USTEC_x100m", "USTECm", "USTEC_x100", "USTEC", "NDXUSD#", "NDXUSD"],
    "DOW": ["US30_x100m", "US30m", "US30_x10m", "US30", "DJI"]
}

# ═══════════ HELPERS ═══════════
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
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                account_id = r.json().get("accountId")
                print("Session opened: " + str(account_id))
                return account_id
            else:
                print("Session failed: " + str(r.status_code))
                if attempt < max_retries:
                    time.sleep(5)
        except Exception as e:
            print("Session error: " + str(e))
            if attempt < max_retries:
                time.sleep(5)
    return None

def detect_symbol(account_id, target_idx):
    url = BASE_URL + "/v1/accounts/" + account_id + "/symbols"
    headers = {"Authorization": "Bearer " + TICKERALL_API_KEY}
    try:
        r = requests.get(url, headers=headers, timeout=60)
        data = r.json()
        syms = data.get("symbols", data.get("watched", []))
        
        for candidate in SYMBOL_MAP[target_idx]:
            if candidate in syms:
                return candidate
                
        for s in syms:
            su = str(s).upper()
            if target_idx == "NASDAQ" and ("USTEC" in su or "NDX" in su or "NAS" in su):
                return str(s)
            if target_idx == "DOW" and ("US30" in su or "DJI" in su or "DOW" in su):
                return str(s)
    except Exception as e:
        print(f"detect_symbol {target_idx} error: " + str(e))
    return None

def fetch_candles(account_id, symbol, timeframe, limit=500):
    url = BASE_URL + "/v1/accounts/" + account_id + "/candles"
    headers = {"Authorization": "Bearer " + TICKERALL_API_KEY}
    params = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=90)
        if r.status_code != 200:
            return None
        data = r.json()
        candles = data.get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df[["datetime", "open", "high", "low", "close"]].copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception as e:
        return None

# ═══════════ STATE ═══════════
def get_fresh_state():
    return {
        "cntTotal": 0, "cntTP1": 0, "cntTP2": 0, "cntTP3": 0, "cntSL": 0, "cntPts": 0.0,
        "aOn": [False]*N_MODELS, "aLong": [False]*N_MODELS, "aE": [None]*N_MODELS,
        "aSL": [None]*N_MODELS, "aAnc": [None]*N_MODELS, "aT1": [None]*N_MODELS,
        "aT2": [None]*N_MODELS, "aT3": [None]*N_MODELS, "aRk": [None]*N_MODELS,
        "aH1": [False]*N_MODELS, "aH2": [False]*N_MODELS, "aBarI": [0]*N_MODELS,
        "aLast": [-10000]*N_MODELS, "aTr": [0]*N_MODELS, "aWn": [0]*N_MODELS,
        "aTim": [""]*N_MODELS, "last_bar": None
    }

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            if "NASDAQ" not in s:
                return {"NASDAQ": get_fresh_state(), "DOW": get_fresh_state()}
            return s
        except:
            pass
    return {"NASDAQ": get_fresh_state(), "DOW": get_fresh_state()}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

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
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    
    for i in range(left, n - right):
        if highs[i] == max(highs[i-left:i+right+1]) and highs[i] > max(highs[i-left:i]):
            sh[i+right] = highs[i]
        if lows[i] == min(lows[i-left:i+right+1]) and lows[i] < min(lows[i-left:i]):
            sl[i+right] = lows[i]
            
    df["sh"] = sh
    df["sl"] = sl
    df["lastSH"] = df["sh"].ffill()
    df["lastSL"] = df["sl"].ffill()
    return df

# ═══════════ SESSIONS ═══════════
def check_sessions(dt_ny):
    hm = dt_ny.hour * 100 + dt_ny.minute
    londonO = 200 <= hm < 300
    sbLDN = 300 <= hm < 400
    judas = 930 <= hm < 1000
    sbAM = 1000 <= hm < 1100
    y2022 = 1100 <= hm < 1130
    lunch = 1150 <= hm < 1210
    sbPM = 1400 <= hm < 1500
    moc = 1515 <= hm < 1545
    fomc = 1400 <= hm < 1430
    nfp = 830 <= hm < 900
    lunchBl = 1210 <= hm < 1359
    isFri = dt_ny.weekday() == 4
    tgif = isFri and (1400 <= hm < 1500)
    amKZ = 930 <= hm < 1130
    
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
    
    recentLow = df_m30["low"].iloc[i-20:i].min()
    recentHigh = df_m30["high"].iloc[i-20:i].max()
    
    bullSweepOK = False
    bullSweepLow = None
    for j in range(max(0, i-20), i+1):
        r = df_m30.iloc[j]
        rl = df_m30["low"].iloc[max(0, j-20):j].min()
        if r["low"] < rl and r["close"] > rl:
            bullSweepOK = True
            bullSweepLow = r["low"]
            
    bearSweepOK = False
    bearSweepHigh = None
    for j in range(max(0, i-20), i+1):
        r = df_m30.iloc[j]
        rh = df_m30["high"].iloc[max(0, j-20):j].max()
        if r["high"] > rh and r["close"] < rh:
            bearSweepOK = True
            bearSweepHigh = r["high"]

    bullMSS = False
    bearMSS = False
    for j in range(max(0, i-15), i+1):
        r = df_m30.iloc[j]
        p = df_m30.iloc[j-1]
        if not pd.isna(r["lastSH"]) and r["close"] > r["lastSH"] and p["close"] <= r["lastSH"]:
            bullMSS = True
        if not pd.isna(r["lastSL"]) and r["close"] < r["lastSL"] and p["close"] >= r["lastSL"]:
            bearMSS = True

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

    bSig = [False]*5; sSig = [False]*5
    bEA = [None]*5; bSA = [None]*5
    sEA = [None]*5; sSA = [None]*5

    if sessKZ and strongBull and bMitActive and (i - bMitBar) <= 8 and row["low"] <= bMitCE and row["low"] >= bMitLow and row["close"] > row["open"] and row["close"] > bMitCE and (row["close"]-row["open"]) > atr*0.4:
        bSig[0] = True; bEA[0] = bMitCE; bSA[0] = bMitLow - atr*0.3
    if sessKZ and strongBear and sMitActive and (i - sMitBar) <= 8 and row["high"] >= sMitCE and row["high"] <= sMitHigh and row["close"] < row["open"] and row["close"] < sMitCE and (row["open"]-row["close"]) > atr*0.4:
        sSig[0] = True; sEA[0] = sMitCE; sSA[0] = sMitHigh + atr*0.3

    if sessKZ and bearSweepOK and bullMSS and bActive and row["low"] <= bCE and row["close"] > bCE and (row["close"]-row["open"]) > atr*0.5:
        bSig[1] = True; bEA[1] = bCE; bSA[1] = bBot - atr*0.3
    if sessKZ and bullSweepOK and bearMSS and sActive and row["high"] >= sCE and row["close"] < sCE and (row["open"]-row["close"]) > atr*0.5:
        sSig[1] = True; sEA[1] = sCE; sSA[1] = sTop + atr*0.3

    if sessKZ and strongBull and bOBActive and (i - bOBBar) <= 10 and bullMSS and swingEq and bOBMT < swingEq and row["low"] <= bOBMT and row["low"] >= bOBLow and row["close"] > row["open"] and row["close"] > bOBMT:
        bSig[2] = True; bEA[2] = bOBMT; bSA[2] = bOBLow - atr*0.3
    if sessKZ and strongBear and sOBActive and (i - sOBBar) <= 10 and bearMSS and swingEq and sOBMT > swingEq and row["high"] >= sOBMT and row["high"] <= sOBHigh and row["close"] < row["open"] and row["close"] < sOBMT:
        sSig[2] = True; sEA[2] = sOBMT; sSA[2] = sOBHigh + atr*0.3

    if amKZ and bullSweepOK and bullMSS and bOBActive and row["low"] <= bOBMT and row["low"] >= bOBLow and swingEq and bOBMT < swingEq and trendUp:
        bSig[3] = True; bEA[3] = bOBMT; bSA[3] = bOBLow - atr*0.3

    bBodySize = abs(row["close"]-row["open"])
    if sessKZ and strongBull and bActive and bFVG_Age <= 5 and row["low"] <= bCE and row["close"] > bCE and row["close"] > bBot and bBodySize > atr*0.3:
        bSig[4] = True; bEA[4] = bCE; bSA[4] = bBot - atr*0.3
    if sessKZ and strongBear and sActive and sFVG_Age <= 5 and row["high"] >= sCE and row["close"] < sCE and row["close"] < sTop and bBodySize > atr*0.3:
        sSig[4] = True; sEA[4] = sCE; sSA[4] = sTop + atr*0.3

    bar_idx = i
    time_str = fmt_mosul(now_utc)
    
    # Manage Active Trades
    for m in range(5):
        if state["aOn"][m]:
            isL = state["aLong"][m]
            e = state["aE"][m]; slv = state["aSL"][m]; anc = state["aAnc"][m]
            t1 = state["aT1"][m]; t2 = state["aT2"][m]; t3 = state["aT3"][m]
            h1_hit = state["aH1"][m]; h2_hit = state["aH2"][m]
            eb = state["aBarI"][m]
            done = False
            
            if isL:
                if not h1_hit and row["high"] >= t1:
                    h1_hit = True; state["aH1"][m] = True; state["aSL"][m] = e
                    state["cntTP1"] += 1; state["aWn"][m] += 1
                    send_telegram(f"🎯 TP1 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP1: {t1:.2f}\nMove SL to BE: {e:.2f}")
                if h1_hit and not h2_hit and row["high"] >= t2:
                    h2_hit = True; state["aH2"][m] = True
                    state["cntTP2"] += 1
                    send_telegram(f"🎯🎯 TP2 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP2: {t2:.2f}")
                if h2_hit and row["high"] >= t3:
                    state["cntTP3"] += 1; state["cntPts"] += TP3_PTS
                    state["aOn"][m] = False; done = True
                    send_telegram(f"🎯🎯🎯 TP3 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP3: {t3:.2f}\nTrade Closed.")
                
                if not done and USE_INV and not h1_hit and row["close"] < anc:
                    state["cntSL"] += 1
                    state["cntPts"] += ((row["close"] - e) * MULT)
                    state["aOn"][m] = False; done = True
                    send_telegram(f"⚠️ Invalidation Exit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nExit: {row['close']:.2f}")
                    
                if not done and row["low"] <= state["aSL"][m]:
                    if h1_hit:
                        state["cntPts"] += (TP2_PTS if h2_hit else TP1_PTS)
                        send_telegram(f"⚪ SL Hit (Breakeven) [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}")
                    else:
                        state["cntSL"] += 1
                        state["cntPts"] += ((state["aSL"][m] - e) * MULT)
                        send_telegram(f"🛑 SL Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}")
                    state["aOn"][m] = False; done = True
                    
            else: # SELL
                if not h1_hit and row["low"] <= t1:
                    h1_hit = True; state["aH1"][m] = True; state["aSL"][m] = e
                    state["cntTP1"] += 1; state["aWn"][m] += 1
                    send_telegram(f"🎯 TP1 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP1: {t1:.2f}\nMove SL to BE: {e:.2f}")
                if h1_hit and not h2_hit and row["low"] <= t2:
                    h2_hit = True; state["aH2"][m] = True
                    state["cntTP2"] += 1
                    send_telegram(f"🎯🎯 TP2 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP2: {t2:.2f}")
                if h2_hit and row["low"] <= t3:
                    state["cntTP3"] += 1; state["cntPts"] += TP3_PTS
                    state["aOn"][m] = False; done = True
                    send_telegram(f"🎯🎯🎯 TP3 Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nTP3: {t3:.2f}\nTrade Closed.")
                    
                if not done and USE_INV and not h1_hit and row["close"] > anc:
                    state["cntSL"] += 1
                    state["cntPts"] += ((e - row["close"]) * MULT)
                    state["aOn"][m] = False; done = True
                    send_telegram(f"⚠️ Invalidation Exit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nExit: {row['close']:.2f}")
                    
                if not done and row["high"] >= state["aSL"][m]:
                    if h1_hit:
                        state["cntPts"] += (TP2_PTS if h2_hit else TP1_PTS)
                        send_telegram(f"⚪ SL Hit (Breakeven) [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}")
                    else:
                        state["cntSL"] += 1
                        state["cntPts"] += ((e - state["aSL"][m]) * MULT)
                        send_telegram(f"🛑 SL Hit [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nSL: {state['aSL'][m]:.2f}")
                    state["aOn"][m] = False; done = True
                    
            if not done and (bar_idx - eb) >= MAX_BARS_TRADE:
                pv_time = (TP2_PTS if h2_hit else TP1_PTS) if h1_hit else ((row["close"] - e)*MULT if isL else (e - row["close"])*MULT)
                state["cntPts"] += pv_time
                state["aOn"][m] = False; done = True
                send_telegram(f"⏱ Time Exit (60 bars) [{idx_name}] [{NAMES[m]}]\nEntry: {e:.2f}\nCurrent: {row['close']:.2f}")

    # New Signals
    for m in range(5):
        if not state["aOn"][m] and (bar_idx - state["aLast"][m]) > COOLDOWN:
            if bSig[m]:
                e0 = bEA[m]; s0 = bSA[m] - (SL_BUF_PTS / MULT)
                d = abs(e0 - s0) * MULT
                if MIN_SL_PTS <= d <= MAX_SL_PTS:
                    state["aOn"][m] = True; state["aLong"][m] = True
                    state["aE"][m] = e0; state["aSL"][m] = s0
                    state["aAnc"][m] = s0 + (SL_BUF_PTS / MULT) + atr*0.3
                    state["aT1"][m] = e0 + (TP1_PTS / MULT); state["aT2"][m] = e0 + (TP2_PTS / MULT); state["aT3"][m] = e0 + (TP3_PTS / MULT)
                    state["aRk"][m] = d; state["aH1"][m] = False; state["aH2"][m] = False
                    state["aBarI"][m] = bar_idx; state["aLast"][m] = bar_idx
                    state["aTr"][m] += 1; state["cntTotal"] += 1; state["aTim"][m] = time_str
                    
                    msg = f"🟢 BUY [{idx_name}]\nModel: {NAMES[m]}\nEntry: {e0:.2f}\nSL: {s0:.2f}\nTP1: {e0 + TP1_PTS:.2f}\nTP2: {e0 + TP2_PTS:.2f}\nTP3: {e0 + TP3_PTS:.2f}\nPoints: {TP1_PTS:.2f}\n🕐 {time_str}"
                    send_telegram(msg)
            elif sSig[m]:
                e0 = sEA[m]; s0 = sSA[m] + (SL_BUF_PTS / MULT)
                d = abs(s0 - e0) * MULT
                if MIN_SL_PTS <= d <= MAX_SL_PTS:
                    state["aOn"][m] = True; state["aLong"][m] = False
                    state["aE"][m] = e0; state["aSL"][m] = s0
                    state["aAnc"][m] = s0 - (SL_BUF_PTS / MULT) - atr*0.3
                    state["aT1"][m] = e0 - (TP1_PTS / MULT); state["aT2"][m] = e0 - (TP2_PTS / MULT); state["aT3"][m] = e0 - (TP3_PTS / MULT)
                    state["aRk"][m] = d; state["aH1"][m] = False; state["aH2"][m] = False
                    state["aBarI"][m] = bar_idx; state["aLast"][m] = bar_idx
                    state["aTr"][m] += 1; state["cntTotal"] += 1; state["aTim"][m] = time_str
                    
                    msg = f"🔴 SELL [{idx_name}]\nModel: {NAMES[m]}\nEntry: {e0:.2f}\nSL: {s0:.2f}\nTP1: {e0 - TP1_PTS:.2f}\nTP2: {e0 - TP2_PTS:.2f}\nTP3: {e0 - TP3_PTS:.2f}\nPoints: {TP1_PTS:.2f}\n🕐 {time_str}"
                    send_telegram(msg)

    return state

# ═══════════ MAIN LOOP ═══════════
def main():
    print("ICT NASDAQ & DOW Elite Bot - start")
    state = load_state()
    now_utc = datetime.now(ZoneInfo("UTC"))
    now_ny = now_utc.astimezone(NY_TZ)

    if now_ny.weekday() >= 5:
        print("Weekend - disabled")
        return

    account_id = open_session()
    if not account_id:
        return

    for idx_name in ["NASDAQ", "DOW"]:
        print(f"--- Checking {idx_name} ---")
        sym = detect_symbol(account_id, idx_name)
        if not sym:
            print(f"Symbol not found for {idx_name}")
            continue
            
        print(f"Using symbol: {sym}")

        df_m30 = fetch_candles(account_id, sym, "M30", 500)
        df_h1 = fetch_candles(account_id, sym, "H1", 500)
        
        if df_m30 is None or df_h1 is None:
            print(f"Failed to fetch data for {idx_name}")
            continue

        last_bar_time = str(df_m30.iloc[-1]["datetime"])
        if state[idx_name].get("last_bar") == last_bar_time:
            print(f"{idx_name} bar already processed.")
            continue
        
        state[idx_name]["last_bar"] = last_bar_time
        state[idx_name] = check_strategy(df_m30, df_h1, state[idx_name], now_utc, idx_name)
        
        s = state[idx_name]
        print(f"{idx_name} Stats -> Trades: {s['cntTotal']} | TP1: {s['cntTP1']} | SL: {s['cntSL']} | Points: {s['cntPts']:.2f}")

    save_state(state)

if __name__ == "__main__":
    main()