
# /content/src/ingestion/simulator.py  (v5: schema v4 with driver history, vehicle metadata, environment indices)
import os, uuid, time, random, json, argparse, logging, hashlib
import numpy as np, pandas as pd, yaml
from fastavro import writer, parse_schema

def setup_logger(level: str = "INFO"):
    level = level.upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    logging.getLogger("fastavro").setLevel(logging.WARNING)

def load_config(config_path, schema_path):
    with open(config_path, 'r') as f: cfg = yaml.safe_load(f)
    with open(schema_path, 'r') as f: sch = json.load(f)
    return cfg, parse_schema(sch)

def validate_config(cfg: dict):
    required_top = ["seed","num_drivers","events_per_second","trip","geofence","output"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"Missing required config key: {k}")

# === Avro-safe enum values ===
AGE_BANDS = ["UNDER25","AGE_25_40","AGE_40_60","OVER60"]
VEHICLE_MAKES = ["Toyota","Ford","Honda","Tesla","BMW","Chevrolet"]
VEHICLE_TYPES = ["SEDAN","SUV","TRUCK","MOTORCYCLE","EV","OTHER"]
OWNERSHIP_TYPES = ["OWNED","LEASED","FLEET"]
ROAD_TYPES = ["highway","urban","rural"]
WEATHER_TYPES = ["clear","rain","snow","fog","other"]

# --- Context generators ---
def assign_driver_context(driver_id, seed_offset=0):
    random.seed(hash(driver_id) + seed_offset)
    age_band = random.choice(AGE_BANDS)

    # license years depends loosely on age band
    if age_band == "UNDER25":
        license_years = random.randint(1, 7)
    elif age_band == "AGE_25_40":
        license_years = random.randint(5, 20)
    elif age_band == "AGE_40_60":
        license_years = random.randint(10, 35)
    else:  # OVER60
        license_years = random.randint(20, 50)

    # driver history
    accidents_count = random.randint(0, 3)
    claims_count = random.randint(0, 5)
    violation_points = random.randint(0, 12)
    policy_tenure_years = random.randint(0, 20)

    return {
        "driver_age_band": age_band,
        "license_years": license_years,
        "accidents_count": accidents_count,
        "claims_count": claims_count,
        "violation_points": violation_points,
        "policy_tenure_years": policy_tenure_years,
    }

def assign_vehicle_context(driver_id, seed_offset=0):
    random.seed(hash(driver_id) + seed_offset)
    vehicle_make = random.choice(VEHICLE_MAKES)
    vehicle_model_year = random.randint(2005, 2024)
    safety_rating = round(random.uniform(2.5, 5.0), 1)
    vehicle_type = random.choice(VEHICLE_TYPES)
    ownership_type = random.choice(OWNERSHIP_TYPES)
    # hash driver_id+seed to make stable VIN hash
    vin_hash = hashlib.sha1(f"{driver_id}-{seed_offset}".encode()).hexdigest()[:12]

    return {
        "vehicle_make": vehicle_make,
        "vehicle_model_year": vehicle_model_year,
        "safety_rating": safety_rating,
        "vehicle_vin_hash": vin_hash,
        "vehicle_type": vehicle_type,
        "ownership_type": ownership_type,
    }

def assign_trip_context(seed_offset=0):
    random.seed(time.time() + seed_offset)
    weather = random.choice(WEATHER_TYPES)
    road_type = random.choice(ROAD_TYPES)
    posted_speed_limit = random.choice([30, 40, 50, 60, 70, 80, 100, 120])
    crime_index = round(random.uniform(0, 10), 2)      # proxy crime risk
    accident_index = round(random.uniform(0, 10), 2)   # proxy accident density

    return {
        "weather": weather,
        "road_type": road_type,
        "posted_speed_limit": posted_speed_limit,
        "crime_index": crime_index,
        "accident_index": accident_index,
    }

# --- Simulator ---
def simulate_trip(cfg, driver_id, trip_id, start_ts_ms, driver_ctx, vehicle_ctx, trip_ctx):
    n_minutes = random.randint(cfg["trip"]["min_minutes"], cfg["trip"]["max_minutes"])
    hz = cfg["events_per_second"]
    n_events = n_minutes * 60 * hz

    base_speed, std = cfg["trip"]["avg_speed_mps"], cfg["trip"]["speed_stddev"]
    lat_min, lat_max = cfg["geofence"]["lat_min"], cfg["geofence"]["lat_max"]
    lon_min, lon_max = cfg["geofence"]["lon_min"], cfg["geofence"]["lon_max"]

    lat = float(np.random.uniform(lat_min, lat_max))
    lon = float(np.random.uniform(lon_min, lon_max))
    ts = int(start_ts_ms)
    odometer = float(np.random.uniform(1000, 100000))
    heading = float(np.random.uniform(0, 360))

    events = []
    for seq_no in range(n_events):
        speed = max(0.0, float(np.random.normal(base_speed, std)))
        accel = float(np.random.normal(0, 0.5))
        if np.random.rand() < cfg["trip"]["harsh_brake_prob"]:
            accel = float(np.random.uniform(-5, -3))
        elif np.random.rand() < cfg["trip"]["harsh_accel_prob"]:
            accel = float(np.random.uniform(3, 5))

        gyro = float(np.random.normal(0, 0.1))
        heading = (heading + float(np.random.normal(0, 5))) % 360.0
        lat += float(np.random.normal(0, 0.00005))
        lon += float(np.random.normal(0, 0.00005))
        odometer += speed / hz / 1000.0

        events.append({
            "event_id": str(uuid.uuid4()),
            "driver_id": driver_id,
            "trip_id": trip_id,
            "ts": ts,
            "ingestion_ts": int(time.time() * 1000),
            "event_seq_no": seq_no,
            "lat": lat,
            "lon": lon,
            "speed_mps": speed,
            "accel_mps2": accel,
            "gyro": gyro,
            "heading": heading,
            "odometer_km": odometer,
            "source": "simulator",
            **driver_ctx,
            **vehicle_ctx,
            **trip_ctx
        })
        ts += int(round(1000.0 / hz))
    return events

def run(config_path, schema_path, batch_size=250, log_level="INFO"):
    setup_logger(log_level)
    logger = logging.getLogger(__name__)

    cfg, parsed_schema = load_config(config_path, schema_path)
    validate_config(cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    out_base = cfg["output"]["path"]
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    now_ms = int(time.time() * 1000)
    total_events = 0
    batch, bidx = [], 0

    logger.info("start_run config=%s schema=%s num_drivers=%s", config_path, schema_path, cfg["num_drivers"])

    for i in range(cfg["num_drivers"]):
        random.seed(cfg["seed"] + i)
        np.random.seed(cfg["seed"] + i)

        driver_id = f"driver_{i:05d}"
        trip_id = str(uuid.uuid4())
        start_ts_ms = now_ms - int(np.random.uniform(0, 24*3600*1000))

        driver_ctx = assign_driver_context(driver_id, seed_offset=i)
        vehicle_ctx = assign_vehicle_context(driver_id, seed_offset=i)
        trip_ctx = assign_trip_context(seed_offset=i)

        batch.extend(simulate_trip(cfg, driver_id, trip_id, start_ts_ms, driver_ctx, vehicle_ctx, trip_ctx))

        if (i+1) % batch_size == 0 or (i+1) == cfg["num_drivers"]:
            try:
                df = pd.DataFrame(batch)
                if "parquet" in cfg.get("output", {}).get("format", []):
                    pq = f"{out_base}_batch{bidx}.parquet"
                    df.to_parquet(pq, index=False, engine="pyarrow")
                if "avro" in cfg.get("output", {}).get("format", []):
                    av = f"{out_base}_batch{bidx}.avro"
                    with open(av, "wb") as f:
                        writer(f, parsed_schema, batch)

                total_events += len(df)
                logger.info("write_batch batch=%s events=%s total_events=%s", bidx, len(df), total_events)
                bidx += 1
                batch = []
            except Exception as e:
                logger.exception("batch_write_failed batch=%s error=%s", bidx, repr(e))
                raise
    logger.info("end_run total_events=%s batches=%s", total_events, bidx)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--schema', required=True)
    p.add_argument('--batch', type=int, default=250)
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args()
    run(args.config, args.schema, batch_size=args.batch, log_level=args.log_level)
