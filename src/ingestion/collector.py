
# /content/src/ingestion/collector.py
# Collector API — Phase 2 Step 2b (Unified)
# - Schema v4 support (/docs/schemas/telematics_event_v4.avsc)
# - Extended validation (driver/vehicle/environment context)
# - Bronze + DLQ persistence
# - HMAC authentication
# - Idempotency (LRU cache of event_ids)
# - Rate limiting per driver

import os, json, uuid, logging, time, hmac, hashlib, yaml
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastavro import parse_schema, validation
from collections import OrderedDict, defaultdict, deque

# ---------- Paths ----------
SCHEMA_PATH   = "docs/schemas/telematics_event_v4.avsc"
AUTH_CFG_PATH = "config/collector_auth.yaml"
BRONZE_DIR    = "data/bronze"
DLQ_FILE      = "data/dlq/events_dlq.jsonl"

os.makedirs(BRONZE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DLQ_FILE), exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("collector")

# ---------- Load Schema ----------
with open(SCHEMA_PATH) as f:
    schema_dict = json.load(f)
parsed_schema = parse_schema(schema_dict)

# ---------- Load Device Secrets ----------
if os.path.exists(AUTH_CFG_PATH):
    with open(AUTH_CFG_PATH) as f:
        AUTH_CFG = yaml.safe_load(f)
else:
    AUTH_CFG = {"devices": {}}
DEVICE_KEYS = AUTH_CFG.get("devices", {})

# ---------- Buffers & Metrics ----------
RAW_TOPIC, DLQ_BUF = [], []
bronze_batch_idx = 0
BATCH_SIZE = 50
metrics = {
    "accepted": 0,
    "rejected": 0,
    "unauth": 0,
    "duplicates": 0,
    "rate_limited": 0,
}

# ---------- Idempotency: LRU cache ----------
class LRUCache:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.data = OrderedDict()
    def add(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return False
        self.data[key] = True
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return True

event_cache = LRUCache(capacity=10000)

# ---------- Rate Limiter ----------
driver_windows = defaultdict(lambda: deque())  # driver_id -> timestamps

# ---------- Load Collector Config (rate limit, etc.) ----------
CFG = {}
if os.path.exists("config/collector.yaml"):
    import yaml
    with open("config/collector.yaml") as f:
        CFG = yaml.safe_load(f) or {}

rl_cfg = CFG.get("rate_limit", {})
MAX_EVENTS_PER_SEC = int(rl_cfg.get("max_events_per_sec", 20))
WINDOW_SEC = float(rl_cfg.get("window_sec", 1.0))

logger.info("Rate limiter set: max_events_per_sec=%s window_sec=%s",
            MAX_EVENTS_PER_SEC, WINDOW_SEC)

def check_rate_limit(driver_id: str) -> bool:
    now = time.time()
    dq = driver_windows[driver_id]
    while dq and dq[0] < now - WINDOW_SEC:
        dq.popleft()
    if len(dq) >= MAX_EVENTS_PER_SEC:
        return False
    dq.append(now)
    return True

# ---------- Helpers ----------
def flush_raw():
    global RAW_TOPIC, bronze_batch_idx
    if not RAW_TOPIC:
        return
    df = pd.DataFrame(RAW_TOPIC)
    out_file = os.path.join(BRONZE_DIR, f"bronze_batch{bronze_batch_idx}.parquet")
    df.to_parquet(out_file, index=False, engine="pyarrow")
    logger.info("flushed_bronze file=%s rows=%s", out_file, len(df))
    bronze_batch_idx += 1
    RAW_TOPIC = []

def flush_dlq():
    global DLQ_BUF
    if not DLQ_BUF:
        return
    with open(DLQ_FILE, "a") as f:
        for ev in DLQ_BUF:
            f.write(json.dumps(ev) + "\n")
    logger.info("flushed_dlq file=%s added=%s", DLQ_FILE, len(DLQ_BUF))
    DLQ_BUF = []

def nullsafe_float(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) else None

# ---------- Extended Validation ----------
VALID_AGE_BANDS   = {"UNDER25", "AGE_25_40", "AGE_40_60", "OVER60"}
VALID_VEHICLE_TYPES = {"SEDAN", "SUV", "TRUCK", "MOTORCYCLE", "EV", "OTHER"}
VALID_OWNERSHIP   = {"OWNED", "LEASED", "FLEET"}
VALID_WEATHER     = {"clear", "rain", "snow", "fog", "other"}
VALID_ROAD        = {"highway", "urban", "rural"}

def validate_event(ev: Dict[str, Any]) -> Tuple[bool, str]:
    # Schema-level validation
    try:
        if not validation.validate(ev, parsed_schema):
            return False, "SCHEMA_VALIDATION_FAILED"
    except Exception:
        return False, "SCHEMA_VALIDATION_FAILED"

    # Core checks
    lat, lon = nullsafe_float(ev.get("lat")), nullsafe_float(ev.get("lon"))
    spd, acc, hdg = nullsafe_float(ev.get("speed_mps")), nullsafe_float(ev.get("accel_mps2")), nullsafe_float(ev.get("heading"))

    if lat is None or lon is None: return False, "MISSING_LAT_LON"
    if not (-90 <= lat <= 90 and -180 <= lon <= 180): return False, "LAT_LON_OUT_OF_RANGE"
    if spd is None or spd < 0: return False, "NEGATIVE_SPEED"
    if acc is not None and abs(acc) > 10: return False, "ACCEL_OUT_OF_RANGE"
    if hdg is not None and not (0 <= hdg < 360): return False, "HEADING_OUT_OF_RANGE"

    # Extended fields sanity checks
    if ev.get("driver_age_band") and ev["driver_age_band"] not in VALID_AGE_BANDS:
        return False, "INVALID_DRIVER_AGE_BAND"
    if ev.get("vehicle_type") and ev["vehicle_type"] not in VALID_VEHICLE_TYPES:
        return False, "INVALID_VEHICLE_TYPE"
    if ev.get("ownership_type") and ev["ownership_type"] not in VALID_OWNERSHIP:
        return False, "INVALID_OWNERSHIP_TYPE"
    if ev.get("weather") and ev["weather"] not in VALID_WEATHER:
        return False, "INVALID_WEATHER"
    if ev.get("road_type") and ev["road_type"] not in VALID_ROAD:
        return False, "INVALID_ROAD_TYPE"

    # Ranges for numeric extended fields
    if ev.get("safety_rating") is not None and not (0 <= ev["safety_rating"] <= 5):
        return False, "INVALID_SAFETY_RATING"
    if ev.get("vehicle_model_year") is not None and not (1980 <= ev["vehicle_model_year"] <= 2025):
        return False, "INVALID_MODEL_YEAR"
    if ev.get("crime_index") is not None and ev["crime_index"] < 0:
        return False, "INVALID_CRIME_INDEX"
    if ev.get("accident_index") is not None and ev["accident_index"] < 0:
        return False, "INVALID_ACCIDENT_INDEX"

    return True, "OK"

def verify_signature(driver_id: str, raw_body: bytes, signature: str) -> bool:
    secret = DEVICE_KEYS.get(driver_id)
    if not secret:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

# ---------- FastAPI ----------
app = FastAPI(title="Telematics Collector API (Unified v4)")

@app.post("/ingest")
async def ingest_event(request: Request):
    req_id = str(uuid.uuid4())
    raw_body = await request.body()

    try:
        event = json.loads(raw_body.decode("utf-8"))
    except Exception:
        logger.warning("req_id=%s invalid_json", req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON")

    driver_id = event.get("driver_id")
    signature = request.headers.get("X-Signature", "")

    # --- Auth ---
    if not verify_signature(driver_id, raw_body, signature):
        metrics["unauth"] += 1
        logger.warning("req_id=%s unauth driver_id=%s", req_id, driver_id)
        raise HTTPException(status_code=401, detail="Unauthorized or bad signature")

    # --- Idempotency ---
    if not event_cache.add(event["event_id"]):
        metrics["duplicates"] += 1
        return {"status": "rejected", "reason": "DUPLICATE_EVENT"}

    # --- Rate limiting ---
    if not check_rate_limit(driver_id):
        metrics["rate_limited"] += 1
        return {"status": "rejected", "reason": "RATE_LIMIT_EXCEEDED"}

    # --- Validation ---
    ok, reason = validate_event(event)
    if not ok:
        event["_dlq_reason"] = reason
        DLQ_BUF.append(event)
        metrics["rejected"] += 1
        flush_dlq()
        return {"status": "rejected", "reason": reason}

    RAW_TOPIC.append(event)
    metrics["accepted"] += 1
    if len(RAW_TOPIC) >= BATCH_SIZE:
        flush_raw()
    return {"status": "accepted", "queue_size": len(RAW_TOPIC)}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "metrics": metrics,
        "bronze_files": len(os.listdir(BRONZE_DIR)),
        "dlq_file_size": os.path.getsize(DLQ_FILE) if os.path.exists(DLQ_FILE) else 0,
    }

@app.post("/flush")
def flush():
    flush_raw()
    flush_dlq()
    return {"status": "flushed"}
