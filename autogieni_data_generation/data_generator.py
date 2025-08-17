# Synthetic structured logs generator for in-car adaptive preference system
# Produces JSONL logs for:
# - telemetry.signals
# - system.suggestion
# - user.action
# - training.feedback
#
# You can tweak the constants in the CONFIG block to change volume/behavior.

import os
import json
import math
import uuid
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
SEED = 42
NUM_DRIVERS = 8
VEHICLES = [f"VIN{i:03d}" for i in range(1, 6)]
DAYS_PER_DRIVER = 7
GENERATE_DOMAINS = ["HVAC", "Infotainment", "Navigation", "Comfort", "Lighting"]
# For each "episode" (decision point), we log 1 suggestion per domain (if applicable).
# We'll create ~ 2 decision points per day (morning & evening), plus some random leisure ones.

BASE_DATE = datetime(2025, 8, 1, tzinfo=timezone.utc)  # start of the synthetic period (UTC)
OUTPUT_DIR = "/mnt/data/synthetic_logs"

# Reward window (seconds) to count a suggestion as "kept"
KEEP_WINDOW_S = 60

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Helper utilities
# -----------------------------
def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def write_jsonl(path: str, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            # ensure datetimes are serialized
            for k, v in list(r.items()):
                if isinstance(v, datetime):
                    r[k] = iso(v)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    s = e / np.sum(e)
    return s

def geodesic_noise():
    # Lightweight pseudo geohash/pluscode placeholder
    return "LOC_" + uuid.uuid4().hex[:8]

def time_of_day_bucket(dt: datetime) -> str:
    h = dt.hour
    if 6 <= h < 10:
        return "morning"
    if 10 <= h < 16:
        return "midday"
    if 16 <= h < 20:
        return "evening"
    if 20 <= h or h < 6:
        return "night"
    return "unknown"

def dow_str(dt: datetime) -> str:
    return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dt.weekday()]

def season_factor(dt: datetime) -> float:
    # August in many regions is warm; keep simple factor
    return 1.0

def solar_intensity(dt: datetime, clouds: float) -> float:
    # Max around midday; clouds in [0,1] attenuate
    h = dt.hour + dt.minute/60.0
    # model daylight as a raised cosine between 6 and 18
    if h < 6 or h > 18:
        base = 0.0
    else:
        # peak at 12:00 ~ 800 W/m2
        base = 800.0 * math.cos((h-12) * math.pi/12)**2
    return max(0.0, base * (1.0 - 0.7*clouds))

def traffic_level(dt: datetime) -> str:
    tod = time_of_day_bucket(dt)
    if tod == "morning" or tod == "evening":
        return np.random.choice(["high","medium","low"], p=[0.55,0.35,0.10])
    if tod == "midday":
        return np.random.choice(["high","medium","low"], p=[0.15,0.5,0.35])
    return np.random.choice(["high","medium","low"], p=[0.10,0.30,0.60])

def speed_given_traffic(level: str) -> float:
    return {
        "high": np.random.normal(12, 5),
        "medium": np.random.normal(35, 10),
        "low": np.random.normal(60, 15)
    }[level]

def precipitation_today() -> bool:
    return np.random.rand() < 0.25  # 25% chance of rain day

def humidity_from_precip(precip: bool) -> float:
    if precip:
        return float(np.clip(np.random.normal(80, 7), 55, 98))
    else:
        return float(np.clip(np.random.normal(40, 10), 20, 75))

def outside_temp(dt: datetime, precip: bool) -> float:
    # simple diurnal curve: cool mornings, hot afternoons, mild nights
    tod = time_of_day_bucket(dt)
    base = 34 if tod in ("midday","evening") else (26 if tod == "night" else 29)
    # rain cools it a bit
    delta = -3 if precip else 0
    return float(np.clip(np.random.normal(base + delta, 2.5), 10, 45))

def cabin_temp_from_outside(out_c: float, ac_on: bool, setpoint: float, elapsed_s: int) -> float:
    # coarse dynamics: moves toward setpoint if AC on, otherwise toward outside
    tau = 300.0  # ~5 min time constant
    if ac_on:
        target = setpoint
    else:
        target = (0.7*out_c + 0.3*setpoint)
    alpha = 1.0 - math.exp(-elapsed_s/tau)
    return float((1-alpha)*out_c + alpha*target + np.random.normal(0, 0.2))

def prob_from_distance(delta: float, slope: float = 1.4) -> float:
    # logistic acceptance prob that decays with |delta| (in degrees)
    return float(1 / (1 + math.exp(slope*(abs(delta)-0.5))))

# -----------------------------
# Driver profiles
# -----------------------------

@dataclass
class DriverProfile:
    user_id: str
    vehicle_id: str
    cluster: str  # "cool_morning", "warm_evening", etc.
    prefers_recirc: bool
    dislikes_fan_noise: bool
    seat_heat_threshold_c: float
    music_pref: Dict[str, str]  # by time_of_day: {"morning":"news","evening":"pop",...}
    volume_base: int  # 0..10
    destination_habits: Dict[str, List[Tuple[str, float]]]  # by time_of_day: [(place, prob), ...]
    sensitivity: float  # how sharply they reject non-preferred actions
    override_rate_bias: float  # base propensity to override

def make_driver_profiles(n: int) -> List[DriverProfile]:
    clusters = ["cool_mornings", "warm_mornings", "neutral", "very_cool"]
    music_sources = ["news", "podcast", "pop", "classical", "hiphop", "edm"]
    destination_sets = [
        ("work", 0.7), ("gym", 0.1), ("grocery", 0.1), ("daycare", 0.1)
    ]
    profiles = []
    for i in range(n):
        user_id = f"driver_{i+1:03d}"
        vehicle_id = random.choice(VEHICLES)
        cluster = random.choice(clusters)
        prefers_recirc = random.random() < 0.4
        dislikes_fan_noise = random.random() < 0.5
        seat_heat_threshold_c = random.choice([10.0, 12.0, 14.0])
        # time-of-day music prefs
        mp = {
            "morning": np.random.choice(["news","podcast","classical","pop"], p=[0.45,0.25,0.15,0.15]),
            "midday": np.random.choice(["pop","hiphop","edm","podcast","classical"], p=[0.30,0.25,0.20,0.15,0.10]),
            "evening": np.random.choice(["pop","hiphop","podcast","classical","news"], p=[0.35,0.25,0.15,0.15,0.10]),
            "night": np.random.choice(["podcast","classical","news","pop"], p=[0.4,0.3,0.2,0.1])
        }
        volume_base = int(np.clip(np.random.normal(5, 2), 1, 9))
        # destination habits
        dest_morning = [("work", 0.75), ("daycare", 0.15), ("gym", 0.05), ("grocery", 0.05)]
        dest_evening = [("home", 0.7), ("gym", 0.15), ("grocery", 0.1), ("other", 0.05)]
        dest_midday = [("grocery", 0.3), ("other", 0.3), ("gym", 0.2), ("home", 0.2)]
        dest_night = [("home", 0.8), ("other", 0.2)]
        destination_habits = {
            "morning": dest_morning,
            "midday": dest_midday,
            "evening": dest_evening,
            "night": dest_night,
        }
        sensitivity = float(np.clip(np.random.normal(1.2, 0.3), 0.7, 2.0))
        override_rate_bias = float(np.clip(np.random.beta(2,5), 0.05, 0.5))
        profiles.append(DriverProfile(
            user_id=user_id,
            vehicle_id=vehicle_id,
            cluster=cluster,
            prefers_recirc=prefers_recirc,
            dislikes_fan_noise=dislikes_fan_noise,
            seat_heat_threshold_c=seat_heat_threshold_c,
            music_pref=mp,
            volume_base=volume_base,
            destination_habits=destination_habits,
            sensitivity=sensitivity,
            override_rate_bias=override_rate_bias
        ))
    return profiles

DRIVERS = make_driver_profiles(NUM_DRIVERS)

# -----------------------------
# Baseline policy (interpretable heuristics)
# -----------------------------

def hvac_baseline_suggestion(ctx: dict, prof: DriverProfile) -> dict:
    out_c = ctx["outside_temp_c"]
    tod = ctx["time_of_day"]
    solar = ctx["solar_w_m2"]
    traffic = ctx["traffic"]
    humid = ctx["humidity_pct"]
    # cluster-based base setpoint
    base = 21.0
    if prof.cluster == "cool_mornings" and tod == "morning":
        base = 20.0
    elif prof.cluster == "warm_mornings" and tod == "morning":
        base = 22.0
    elif prof.cluster == "very_cool":
        base = 20.0
    # adjust by outside temp
    if out_c >= 32:
        base -= 1.0
    if out_c <= 12:
        base += 1.0
    # limit to [19,23]
    setpoint = float(np.clip(round(base), 19, 23))
    ac_on = bool(out_c >= 22 or humid >= 70)
    recirc = bool(prof.prefers_recirc and traffic in ("high","medium") and out_c >= 24)
    # fan heuristic
    fan = 1
    if solar >= 500 and ctx["speed_kmh"] < 20:
        fan = 3
    elif out_c >= 30:
        fan = 2
    if prof.dislikes_fan_noise and fan == 3:
        fan = 2
    airflow = "auto"
    if humid >= 85:
        airflow = "defog"
    return {
        "setpoint_c": int(setpoint),
        "fan": int(fan),
        "airflow_mode": airflow,
        "ac_on": ac_on,
        "recirculation": recirc
    }

def hvac_candidate_neighbors(sugg: dict) -> List[dict]:
    # Nearby safe candidates (±1 step changes)
    cands = []
    for d in [-1,0,1]:
        for df in [-1,0,1]:
            sp = int(np.clip(sugg["setpoint_c"] + d, 19, 23))
            fan = int(np.clip(sugg["fan"] + df, 1, 3))
            cand = dict(sugg)
            cand["setpoint_c"] = sp
            cand["fan"] = fan
            cands.append(cand)
    # remove duplicates
    seen = set()
    uniq = []
    for c in cands:
        key = (c["setpoint_c"], c["fan"], c["airflow_mode"], c["ac_on"], c["recirculation"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def infotainment_baseline(ctx: dict, prof: DriverProfile) -> dict:
    tod = ctx["time_of_day"]
    source = prof.music_pref[tod]
    volume = int(np.clip(np.random.normal(prof.volume_base, 1), 1, 10))
    # simple playlist name
    playlist = {
        "news": "Daily Briefing",
        "podcast": "Top Podcasts",
        "pop": "Pop Mix",
        "classical": "Calm Classical",
        "hiphop": "Hip-Hop Daily",
        "edm": "EDM Now"
    }.get(source, "Favorites")
    return {"source": source, "playlist": playlist, "volume": volume}

def infotainment_candidates(sugg: dict) -> List[dict]:
    # Change volume ±1 or switch between source and playlist variants
    cands = []
    for dv in [-1, 0, 1]:
        vol = int(np.clip(sugg["volume"] + dv, 1, 10))
        cand = dict(sugg)
        cand["volume"] = vol
        cands.append(cand)
    # alternative playlists for the same source
    alt = {
        "news": ["Daily Briefing","Morning Update","Global News"],
        "podcast": ["Top Podcasts","Tech Today","Thoughtful Talks"],
        "pop": ["Pop Mix","Fresh Pop","Top Hits"],
        "classical": ["Calm Classical","Baroque Focus","Piano Essentials"],
        "hiphop": ["Hip-Hop Daily","Rap Radar","Street Beats"],
        "edm": ["EDM Now","Dance Floor","Deep House"]
    }
    for p in alt.get(sugg["source"], []):
        cand = dict(sugg)
        cand["playlist"] = p
        cands.append(cand)
    # dedupe
    seen = set()
    uniq = []
    for c in cands:
        key = (c["source"], c["playlist"], c["volume"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def navigation_baseline(ctx: dict, prof: DriverProfile) -> dict:
    tod = ctx["time_of_day"]
    # choose destination by weighted habit distribution
    options, probs = zip(*prof.destination_habits[tod])
    dest = np.random.choice(options, p=np.array(probs)/np.sum(probs))
    # top-k shortlist: sample 3 with bias
    space = list(options)
    weights = np.array(probs)/np.sum(probs)
    idxs = np.argsort(-weights)[:min(3, len(space))]
    shortlist = [space[i] for i in idxs]
    return {"topk": shortlist, "auto_select": bool(weights[idxs[0]] >= 0.75), "predicted": dest}

def navigation_candidates(sugg: dict) -> List[dict]:
    # Reorder or swap last item
    base = sugg["topk"]
    cands = []
    cands.append({"topk": base, "auto_select": sugg["auto_select"], "predicted": base[0]})
    if len(base) >= 2:
        cands.append({"topk": [base[1]] + [base[0]] + base[2:], "auto_select": False, "predicted": base[1]})
    return cands

def comfort_baseline(ctx: dict, prof: DriverProfile) -> dict:
    out_c = ctx["outside_temp_c"]
    tod = ctx["time_of_day"]
    seat = 1 if out_c <= prof.seat_heat_threshold_c else 0
    if tod == "night" and out_c <= prof.seat_heat_threshold_c + 2:
        seat = min(2, seat + 1)
    steering = 1 if seat >= 1 else 0
    return {"seat_heat": seat, "steering_wheel_heat": steering}

def comfort_candidates(sugg: dict) -> List[dict]:
    cands = []
    for ds in [-1,0,1]:
        sh = int(np.clip(sugg["seat_heat"] + ds, 0, 2))
        cand = dict(sugg)
        cand["seat_heat"] = sh
        cand["steering_wheel_heat"] = 1 if sh >= 1 else 0
        cands.append(cand)
    # dedupe
    seen = set()
    uniq = []
    for c in cands:
        key = (c["seat_heat"], c["steering_wheel_heat"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def lighting_baseline(ctx: dict, prof: DriverProfile) -> dict:
    tod = ctx["time_of_day"]
    solar = ctx["solar_w_m2"]
    brightness = 3  # 1..5
    if tod in ("night",) or solar < 50:
        brightness = 2
    if tod == "night" and solar == 0:
        brightness = 1
    mirror_tilt = "auto"
    return {"ambient_brightness": brightness, "mirror_tilt": mirror_tilt}

def lighting_candidates(sugg: dict) -> List[dict]:
    cands = []
    for db in [-1,0,1]:
        b = int(np.clip(sugg["ambient_brightness"] + db, 1, 5))
        cand = dict(sugg)
        cand["ambient_brightness"] = b
        cands.append(cand)
    return cands

# -----------------------------
# Acceptance models per domain
# -----------------------------

def accept_prob_hvac(ctx, prof, suggestion):
    # preference setpoint target
    preferred = 21.0
    if prof.cluster in ("cool_mornings","very_cool") and ctx["time_of_day"] == "morning":
        preferred = 20.0
    elif prof.cluster == "warm_mornings" and ctx["time_of_day"] == "morning":
        preferred = 22.0
    if ctx["outside_temp_c"] >= 32:
        preferred -= 0.5
    if ctx["outside_temp_c"] <= 12:
        preferred += 0.5
    # acceptance decays with setpoint distance
    p_sp = prob_from_distance(suggestion["setpoint_c"] - preferred, slope=1.3*prof.sensitivity)
    # dislike fan noise
    p_fan = 1.0 if suggestion["fan"] <= 2 or not prof.dislikes_fan_noise else 0.85
    # recirc preference
    p_rec = 1.0 if suggestion["recirculation"] == prof.prefers_recirc else 0.9
    # ac_on helpful when hot/humid
    if (ctx["outside_temp_c"] >= 28 or ctx["humidity_pct"] >= 70) and suggestion["ac_on"]:
        p_ac = 1.0
    elif (ctx["outside_temp_c"] <= 18) and suggestion["ac_on"]:
        p_ac = 0.85
    else:
        p_ac = 0.95
    p = p_sp * p_fan * p_rec * p_ac
    # base override bias
    p *= (1.0 - 0.5*prof.override_rate_bias)
    return float(np.clip(p, 0.01, 0.99))

def accept_prob_infotainment(ctx, prof, suggestion):
    match = 1.0 if suggestion["source"] == prof.music_pref[ctx["time_of_day"]] else 0.7
    vol_penalty = 1.0 - 0.06*abs(suggestion["volume"] - prof.volume_base)
    p = match * max(0.6, vol_penalty)
    p *= (1.0 - 0.4*prof.override_rate_bias)
    return float(np.clip(p, 0.05, 0.99))

def accept_prob_navigation(ctx, prof, suggestion):
    # assume user selects first topk 80% when correct, else overrides to another
    tod = ctx["time_of_day"]
    # pick true destination from profile distribution
    options, probs = zip(*prof.destination_habits[tod])
    true_dest = np.random.choice(options, p=np.array(probs)/np.sum(probs))
    correct_in_topk = true_dest in suggestion["topk"]
    p = 0.85 if correct_in_topk and suggestion["topk"][0] == true_dest else (0.5 if correct_in_topk else 0.2)
    p *= (1.0 - 0.3*prof.override_rate_bias)
    return float(np.clip(p, 0.05, 0.98)), true_dest

def accept_prob_comfort(ctx, prof, suggestion):
    desired = 1 if ctx["outside_temp_c"] <= prof.seat_heat_threshold_c else 0
    if ctx["time_of_day"] == "night" and ctx["outside_temp_c"] <= prof.seat_heat_threshold_c + 2:
        desired = min(2, desired + 1)
    p = 1.0 - 0.3*abs(suggestion["seat_heat"] - desired)
    p *= (1.0 - 0.3*prof.override_rate_bias)
    return float(np.clip(p, 0.1, 0.99))

def accept_prob_lighting(ctx, prof, suggestion):
    # accept minor changes; night prefers lower brightness
    target = 1 if ctx["time_of_day"] == "night" else (2 if ctx["solar_w_m2"] < 50 else 3)
    p = 1.0 - 0.2*abs(suggestion["ambient_brightness"] - target)
    p *= (1.0 - 0.3*prof.override_rate_bias)
    return float(np.clip(p, 0.2, 0.99))

# -----------------------------
# Simulation of episodes
# -----------------------------

def sample_trip_type(dt: datetime) -> str:
    dow = dt.weekday()
    tod = time_of_day_bucket(dt)
    if dow < 5 and tod in ("morning","evening"):
        return "commute"
    return np.random.choice(["leisure","commute","errand"], p=[0.6, 0.2, 0.2])

def simulate_episode(ts: datetime, prof: DriverProfile, trip_id: str, domain: str,
                     telemetry_rows, suggestion_rows, action_rows, feedback_rows):
    # Build context/state snapshot
    precip = precipitation_today()
    hum = humidity_from_precip(precip)
    out_c = outside_temp(ts, precip)
    solar = solar_intensity(ts, clouds=np.random.uniform(0, 1 if not precip else 1.0))
    traffic = traffic_level(ts)
    speed = max(0, speed_given_traffic(traffic))
    occupants = {
        "driver": True,
        "front_passenger": bool(np.random.rand() < 0.2),
        "rear_count": int(np.random.choice([0,1,2], p=[0.7,0.2,0.1]))
    }
    trip_type = sample_trip_type(ts)
    state = {
        "outside_temp_c": round(float(out_c), 1),
        "cabin_temp_c": round(float(np.random.normal(out_c - 2 if out_c>25 else out_c, 1.0)), 1),
        "humidity_pct": round(float(hum), 1),
        "solar_w_m2": round(float(solar), 1),
        "occupants": occupants,
        "trip_type": trip_type,
        "time_of_day": time_of_day_bucket(ts),
        "dow": dow_str(ts),
        "location_hash": geodesic_noise(),
        "speed_kmh": round(float(speed), 1),
        "precipitation": precip,
        "traffic": traffic,
        "calendar_hint": np.random.choice(
            ["none","meeting_9am","pickup_kids","gym_class","grocery_run"],
            p=[0.5,0.2,0.15,0.1,0.05]
        ),
        "recent_overrides_7d": int(np.random.poisson(2))
    }
    telemetry_event = {
        "ts": iso(ts),
        "user_id": prof.user_id,
        "vehicle_id": prof.vehicle_id,
        "trip_id": trip_id,
        "state": state
    }
    telemetry_rows.append(telemetry_event)

    # Domain-specific suggestion and candidates
    policy_version = "policy_v1.0"
    suggestion_id = uuid.uuid4().hex
    candidate_set = []
    selected = None
    confidence = None
    propensity = None
    explanation = ""

    if domain == "HVAC":
        base = hvac_baseline_suggestion(state, prof)
        cands = hvac_candidate_neighbors(base)
        # score candidates with a heuristic acceptance probability to form propensities
        scores = [accept_prob_hvac(state, prof, c) for c in cands]
        probs = softmax(np.array(scores)*3.0)  # sharpened to emphasize better options
        idx = int(np.random.choice(np.arange(len(cands)), p=probs))
        selected = cands[idx]
        candidate_set = cands
        propensity = float(probs[idx])
        confidence = float(np.max(probs))
        explanation = (
            f"Because it's {state['time_of_day']} and outside is {state['outside_temp_c']}°C "
            f"with {state['traffic']} traffic and solar {state['solar_w_m2']} W/m²."
        )
        # acceptance
        p_accept = accept_prob_hvac(state, prof, selected)
        kept = np.random.rand() < p_accept
        # override
        override_within = None
        explicit = None
        if not kept:
            override_within = int(np.random.randint(5, 60))
            # user changes setpoint closer to their preference
            # simple override: +/- 1 step
            ua = dict(selected)
            if selected["setpoint_c"] >= 21:
                ua["setpoint_c"] -= 1
            else:
                ua["setpoint_c"] += 1
            # user action event
            action_rows.append({
                "ts": iso(ts + timedelta(seconds=override_within)),
                "user_id": prof.user_id,
                "trip_id": trip_id,
                "domain": "HVAC",
                "parameter": "bulk",
                "value": ua,
                "origin": "user",
                "suggestion_id": suggestion_id
            })
        # system applies its suggestion as an action event
        action_rows.append({
            "ts": iso(ts),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "HVAC",
            "parameter": "bulk",
            "value": selected,
            "origin": "system",
            "suggestion_id": suggestion_id
        })
        # feedback
        reward = 1.0 if kept else 0.0
        # occasional explicit feedback
        if kept and np.random.rand() < 0.05:
            explicit = "thumbs_up"
        elif (not kept) and np.random.rand() < 0.02:
            explicit = "thumbs_down"
        feedback_rows.append({
            "ts": iso(ts + timedelta(seconds=KEEP_WINDOW_S)),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "HVAC",
            "suggestion_id": suggestion_id,
            "outcome": {
                "kept": kept,
                "override_within_s": override_within,
                "explicit_feedback": explicit
            },
            "reward": reward
        })

    elif domain == "Infotainment":
        base = infotainment_baseline(state, prof)
        cands = infotainment_candidates(base)
        # propensities from acceptance estimates
        scores = [accept_prob_infotainment(state, prof, c) for c in cands]
        probs = softmax(np.array(scores)*3.0)
        idx = int(np.random.choice(np.arange(len(cands)), p=probs))
        selected = cands[idx]
        candidate_set = cands
        propensity = float(probs[idx])
        confidence = float(np.max(probs))
        explanation = (f"{state['time_of_day'].title()} preference: {selected['source']} → {selected['playlist']}.")
        p_accept = accept_prob_infotainment(state, prof, selected)
        kept = np.random.rand() < p_accept
        override_within = None
        explicit = None
        # user might switch source or tweak volume
        if not kept:
            override_within = int(np.random.randint(5, 45))
            ua = dict(selected)
            # switch to preferred source
            ua["source"] = prof.music_pref[state["time_of_day"]]
            ua["playlist"] = "Favorites"
            action_rows.append({
                "ts": iso(ts + timedelta(seconds=override_within)),
                "user_id": prof.user_id,
                "trip_id": trip_id,
                "domain": "Infotainment",
                "parameter": "bulk",
                "value": ua,
                "origin": "user",
                "suggestion_id": suggestion_id
            })
        action_rows.append({
            "ts": iso(ts),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "Infotainment",
            "parameter": "bulk",
            "value": selected,
            "origin": "system",
            "suggestion_id": suggestion_id
        })
        reward = 1.0 if kept else 0.0
        if kept and np.random.rand() < 0.03:
            explicit = "thumbs_up"
        elif (not kept) and np.random.rand() < 0.01:
            explicit = "thumbs_down"
        feedback_rows.append({
            "ts": iso(ts + timedelta(seconds=KEEP_WINDOW_S)),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "Infotainment",
            "suggestion_id": suggestion_id,
            "outcome": {
                "kept": kept,
                "override_within_s": override_within,
                "explicit_feedback": explicit
            },
            "reward": reward
        })

    elif domain == "Navigation":
        base = navigation_baseline(state, prof)
        cands = navigation_candidates(base)
        # propensities proportional to shortlist top candidate strength
        scores = [0.8 if c["predicted"] == c["topk"][0] else 0.6 for c in cands]
        probs = softmax(np.array(scores)*3.0)
        idx = int(np.random.choice(np.arange(len(cands)), p=probs))
        selected = cands[idx]
        candidate_set = cands
        propensity = float(probs[idx])
        confidence = float(np.max(probs))
        explanation = (f"Likely destinations by pattern: {', '.join(selected['topk'])}.")
        p_accept, true_dest = accept_prob_navigation(state, prof, selected)
        kept = np.random.rand() < p_accept
        override_within = None
        explicit = None
        if not kept:
            override_within = int(np.random.randint(5, 50))
            ua = {"chosen": true_dest}
            action_rows.append({
                "ts": iso(ts + timedelta(seconds=override_within)),
                "user_id": prof.user_id,
                "trip_id": trip_id,
                "domain": "Navigation",
                "parameter": "destination",
                "value": ua,
                "origin": "user",
                "suggestion_id": suggestion_id
            })
        action_rows.append({
            "ts": iso(ts),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "Navigation",
            "parameter": "shortlist",
            "value": selected,
            "origin": "system",
            "suggestion_id": suggestion_id
        })
        reward = 1.0 if kept else 0.0
        if kept and selected.get("auto_select") and np.random.rand() < 0.05:
            explicit = "thumbs_up"
        feedback_rows.append({
            "ts": iso(ts + timedelta(seconds=KEEP_WINDOW_S)),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "Navigation",
            "suggestion_id": suggestion_id,
            "outcome": {
                "kept": kept,
                "override_within_s": override_within,
                "explicit_feedback": explicit
            },
            "reward": reward
        })

    elif domain == "Comfort":
        base = comfort_baseline(state, prof)
        cands = comfort_candidates(base)
        scores = [accept_prob_comfort(state, prof, c) for c in cands]
        probs = softmax(np.array(scores)*3.0)
        idx = int(np.random.choice(np.arange(len(cands)), p=probs))
        selected = cands[idx]
        candidate_set = cands
        propensity = float(probs[idx])
        confidence = float(np.max(probs))
        explanation = "Seat/steering heat based on outside temperature and time."
        p_accept = accept_prob_comfort(state, prof, selected)
        kept = np.random.rand() < p_accept
        override_within = None
        explicit = None
        if not kept:
            override_within = int(np.random.randint(5, 40))
            ua = dict(selected)
            ua["seat_heat"] = int(np.clip(ua["seat_heat"] + np.random.choice([-1,1]), 0, 2))
            ua["steering_wheel_heat"] = 1 if ua["seat_heat"] >= 1 else 0
            action_rows.append({
                "ts": iso(ts + timedelta(seconds=override_within)),
                "user_id": prof.user_id,
                "trip_id": trip_id,
                "domain": "Comfort",
                "parameter": "bulk",
                "value": ua,
                "origin": "user",
                "suggestion_id": suggestion_id
            })
        action_rows.append({
            "ts": iso(ts),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "Comfort",
            "parameter": "bulk",
            "value": selected,
            "origin": "system",
            "suggestion_id": suggestion_id
        })
        reward = 1.0 if kept else 0.0
        feedback_rows.append({
            "ts": iso(ts + timedelta(seconds=KEEP_WINDOW_S)),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "Comfort",
            "suggestion_id": suggestion_id,
            "outcome": {
                "kept": kept,
                "override_within_s": override_within,
                "explicit_feedback": None
            },
            "reward": reward
        })

    elif domain == "Lighting":
        base = lighting_baseline(state, prof)
        cands = lighting_candidates(base)
        scores = [accept_prob_lighting(state, prof, c) for c in cands]
        probs = softmax(np.array(scores)*3.0)
        idx = int(np.random.choice(np.arange(len(cands)), p=probs))
        selected = cands[idx]
        candidate_set = cands
        propensity = float(probs[idx])
        confidence = float(np.max(probs))
        explanation = "Ambient brightness adjusted for time and solar intensity."
        p_accept = accept_prob_lighting(state, prof, selected)
        kept = np.random.rand() < p_accept
        override_within = None
        explicit = None
        if not kept:
            override_within = int(np.random.randint(5, 30))
            ua = dict(selected)
            # user tweaks brightness toward target (±1)
            target = 1 if state["time_of_day"] == "night" else (2 if state["solar_w_m2"] < 50 else 3)
            ua["ambient_brightness"] = int(np.clip(
                ua["ambient_brightness"] + np.sign(target - ua["ambient_brightness"]), 1, 5))
            action_rows.append({
                "ts": iso(ts + timedelta(seconds=override_within)),
                "user_id": prof.user_id,
                "trip_id": trip_id,
                "domain": "Lighting",
                "parameter": "bulk",
                "value": ua,
                "origin": "user",
                "suggestion_id": suggestion_id
            })
        action_rows.append({
            "ts": iso(ts),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "Lighting",
            "parameter": "bulk",
            "value": selected,
            "origin": "system",
            "suggestion_id": suggestion_id
        })
        reward = 1.0 if kept else 0.0
        feedback_rows.append({
            "ts": iso(ts + timedelta(seconds=KEEP_WINDOW_S)),
            "user_id": prof.user_id,
            "trip_id": trip_id,
            "domain": "Lighting",
            "suggestion_id": suggestion_id,
            "outcome": {
                "kept": kept,
                "override_within_s": override_within,
                "explicit_feedback": None
            },
            "reward": reward
        })

    # system.suggestion event shared structure
    if selected is not None:
        suggestion_rows.append({
            "ts": iso(ts),
            "user_id": prof.user_id,
            "vehicle_id": prof.vehicle_id,
            "trip_id": trip_id,
            "domain": domain,
            "suggestion_id": suggestion_id,
            "policy_version": policy_version,
            "propensity": propensity,                  # probability assigned to chosen action
            "candidate_count": len(candidate_set),
            "confidence": confidence,
            "candidate_example": candidate_set[0],     # just to illustrate; you could log all candidates elsewhere
            "suggestion": selected,
            "explanation": explanation
        })

def simulate_logs():
    telemetry_rows = []
    suggestion_rows = []
    action_rows = []
    feedback_rows = []

    for prof in DRIVERS:
        # Each driver gets DAYS_PER_DRIVER with 2 fixed decision points (morning, evening) + random midday leisure
        for d in range(DAYS_PER_DRIVER):
            day = BASE_DATE + timedelta(days=d)
            # morning decision ~ 8:15
            morning_ts = day.replace(hour=8, minute=15, second=0)
            # evening decision ~ 18:05
            evening_ts = day.replace(hour=18, minute=5, second=0)
            decision_times = [morning_ts, evening_ts]

            # occasional midday leisure
            if np.random.rand() < 0.6:
                leisure_ts = day.replace(hour=int(np.random.choice([11,12,13,14,15])), minute=int(np.random.choice([0,15,30,45])))
                decision_times.append(leisure_ts)

            # For each decision time, create a "trip_id" and log per-domain suggestions
            for ts in decision_times:
                trip_id = uuid.uuid4().hex[:12]
                # Always simulate HVAC + Navigation; others probabilistically to diversify
                domains = ["HVAC", "Navigation"]
                if np.random.rand() < 0.9:
                    domains.append("Infotainment")
                if np.random.rand() < 0.5:
                    domains.append("Comfort")
                if np.random.rand() < 0.5:
                    domains.append("Lighting")

                for domain in domains:
                    simulate_episode(ts, prof, trip_id, domain,
                                     telemetry_rows, suggestion_rows, action_rows, feedback_rows)

    # Write JSONL files
    write_jsonl(os.path.join(OUTPUT_DIR, "telemetry.signals.jsonl"), telemetry_rows)
    write_jsonl(os.path.join(OUTPUT_DIR, "system.suggestion.jsonl"), suggestion_rows)
    write_jsonl(os.path.join(OUTPUT_DIR, "user.action.jsonl"), action_rows)
    write_jsonl(os.path.join(OUTPUT_DIR, "training.feedback.jsonl"), feedback_rows)

    # Also provide compact preview tables
    df_sugg = pd.DataFrame(suggestion_rows)
    df_feed = pd.DataFrame(feedback_rows)
    df_tel = pd.DataFrame(telemetry_rows)


    return {
        "counts": {
            "telemetry": len(telemetry_rows),
            "suggestions": len(suggestion_rows),
            "actions": len(action_rows),
            "feedback": len(feedback_rows)
        },
        "output_dir": OUTPUT_DIR
    }

result = simulate_logs()
result
