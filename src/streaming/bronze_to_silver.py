import os, glob, yaml, logging, json
import pandas as pd
import numpy as np
from datetime import datetime

# Relative paths - script runs from repo root
BRONZE_DIR = "data/bronze"
CFG_PATH   = "config/etl.yaml"
SCHEMA_PATH = "docs/schemas/telematics_event_v4_silver.avsc"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("bronze_to_silver")

def load_cfg(path=CFG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)

# --- Enum definitions ---
VALID_ENUMS = {
    "driver_age_band": {"UNDER25","AGE_25_40","AGE_40_60","OVER60"},
    "vehicle_type": {"SEDAN","SUV","TRUCK","MOTORCYCLE","EV","OTHER"},
    "ownership_type": {"OWNED","LEASED","FLEET"},
    "weather": {"clear","rain","snow","fog","other"},
    "road_type": {"highway","urban","rural"},
}

def enforce_enums(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """Ensure categorical values are valid according to Silver schema enums."""
    for col, valid_set in VALID_ENUMS.items():
        if col not in df.columns:
            continue
        if drop_invalid:
            before = len(df)
            df = df[df[col].isna() | df[col].isin(valid_set)]
            after = len(df)
            dropped = before - after
            if dropped > 0:
                logger.info("Dropped %s invalid rows for %s", dropped, col)
        else:
            df[col] = df[col].apply(
                lambda x: x if pd.isna(x) or x in valid_set else "OTHER"
            )
    return df

def enrich_trips(df: pd.DataFrame):
    """Aggregate events into trip-level metrics with null-safe handling."""
    agg_funcs = {
        "ts": ["min", "max"],
        "speed_mps": ["mean", "max", "min"],
        "accel_mps2": ["mean"],
        "gyro": ["std"],
        "odometer_km": ["min", "max"]
    }
    grouped = df.groupby(["driver_id", "trip_id"]).agg(agg_funcs)
    grouped.columns = [f"{c[0]}_{c[1]}" for c in grouped.columns]
    grouped = grouped.reset_index()

    grouped["trip_duration_sec"] = (grouped["ts_max"] - grouped["ts_min"]) / 1000.0
    grouped["trip_distance_km"] = grouped["odometer_km_max"] - grouped["odometer_km_min"]
    grouped["avg_speed_calc"] = grouped["trip_distance_km"] / (grouped["trip_duration_sec"] / 3600.0 + 1e-6)

    harsh_brakes = df[df["accel_mps2"] < -3].groupby(["driver_id", "trip_id"]).size()
    harsh_accels = df[df["accel_mps2"] > 3].groupby(["driver_id", "trip_id"]).size()

    grouped = grouped.set_index(["driver_id", "trip_id"])
    grouped["harsh_brakes"] = grouped.index.map(harsh_brakes).fillna(0).astype(int)
    grouped["harsh_accels"] = grouped.index.map(harsh_accels).fillna(0).astype(int)

    df["posted_speed_limit"] = pd.to_numeric(df["posted_speed_limit"], errors="coerce")
    overspeed = df[df["posted_speed_limit"].notna() & (df["speed_mps"] > df["posted_speed_limit"])]
    overspeed_counts = overspeed.groupby(["driver_id", "trip_id"]).size()
    grouped["overspeed_events"] = grouped.index.map(overspeed_counts).fillna(0).astype(int)

    grouped = grouped.reset_index()
    return grouped

def derive_event_date(ts: int) -> str:
    return datetime.utcfromtimestamp(ts/1000.0).strftime("%Y-%m-%d")

def enforce_schema(df: pd.DataFrame, schema_path: str) -> pd.DataFrame:
    with open(schema_path) as f:
        schema = json.load(f)
    schema_fields = [f["name"] for f in schema["fields"]]
    for col in schema_fields:
        if col not in df.columns:
            df[col] = None
    return df[schema_fields]

def clean_bronze():
    cfg = load_cfg()
    thresholds = cfg["thresholds"]
    SILVER_DIR = cfg["output"]["silver_path"]
    partition_on = cfg["output"]["partition_on"]
    write_trip_level = cfg["output"].get("write_trip_level", False)

    os.makedirs(SILVER_DIR, exist_ok=True)

    files = glob.glob(os.path.join(BRONZE_DIR, "*.parquet"))
    if not files:
        logger.warning("No Bronze files found")
        return
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %s raw events from %s Bronze files", len(df), len(files))

    df = df.drop_duplicates(subset=["event_id"])

    df = df[
        df["lat"].between(*thresholds["lat_range"])
        & df["lon"].between(*thresholds["lon_range"])
        & df["speed_mps"].between(thresholds["min_speed_mps"], thresholds["max_speed_mps"])
        & df["accel_mps2"].between(-thresholds["max_accel_mps2"], thresholds["max_accel_mps2"])
    ]

    # Enum enforcement (configurable)
    drop_invalid_enums = cfg["enrichment"].get("drop_invalid_enums", True)
    df = enforce_enums(df, drop_invalid=drop_invalid_enums)

    df = df.sort_values(["driver_id", "trip_id", "ts"]).reset_index(drop=True)
    df["event_date"] = df["ts"].apply(derive_event_date)

    df = enforce_schema(df, SCHEMA_PATH)

    out_file = os.path.join(SILVER_DIR, "silver_events.parquet")
    df.to_parquet(out_file, index=False, engine="pyarrow", partition_cols=partition_on)
    logger.info("✅ Silver events written: %s (%s rows)", out_file, len(df))

    if write_trip_level:
        trip_df = enrich_trips(df)
        trip_file = os.path.join(SILVER_DIR, "silver_trips.parquet")
        trip_df.to_parquet(trip_file, index=False, engine="pyarrow")
        logger.info("✅ Silver trips written: %s (%s rows)", trip_file, len(trip_df))

if __name__ == "__main__":
    clean_bronze()
