
# === Phase 4 : Silver → Gold feature engineering (with schema enforcement) ===
import os, pandas as pd, yaml, logging, json
from datetime import datetime, timezone

CFG_PATH = "config/features.yaml"
GOLD_SCHEMA_PATH = "docs/schemas/driver_features_v1.avsc"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("silver_to_gold")


def load_cfg(path=CFG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def load_gold_schema(path=GOLD_SCHEMA_PATH):
    import json
    with open(path) as f:
        schema = json.load(f)
    return [f["name"] for f in schema["fields"]]


# --- Time helpers ---
from datetime import datetime, timezone
def ts_to_hour(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour

def ts_to_weekday(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).weekday()


def build_features():
    cfg = load_cfg()
    thresholds = cfg["thresholds"]
    SILVER_EVENTS = cfg["input"]["silver_events"]
    SILVER_TRIPS = cfg["input"]["silver_trips"]
    GOLD_DIR = cfg["output"]["gold_dir"]
    os.makedirs(GOLD_DIR, exist_ok=True)

    if not os.path.exists(SILVER_EVENTS):
        logger.error("Silver events file not found: %s", SILVER_EVENTS)
        return

    df = pd.read_parquet(SILVER_EVENTS)
    logger.info("Loaded %s Silver events", len(df))

    # Robust total_km
    total_km_by_driver = None
    if os.path.exists(SILVER_TRIPS):
        trips = pd.read_parquet(SILVER_TRIPS)
        total_km_by_driver = trips.groupby("driver_id", observed=True)["trip_distance_km"].sum()

    # Feature flags
    df["is_harsh_brake"] = df["accel_mps2"] <= thresholds["harsh_brake"]
    df["is_harsh_accel"] = df["accel_mps2"] >= thresholds["harsh_accel"]

    df["hour"] = df["ts"].apply(ts_to_hour)
    night_start, night_end = thresholds["night_start"], thresholds["night_end"]
    df["is_night"] = df["hour"].apply(lambda h: (h >= night_start or h <= night_end))

    df["weekday"] = df["ts"].apply(ts_to_weekday)
    weekend_days = set(cfg["temporal"]["weekend_days"])
    df["is_weekend"] = df["weekday"].isin(weekend_days)

    margin = thresholds.get("overspeed_margin_kmh", 0) / 3.6
    df["is_overspeed"] = df.apply(
        lambda row: False
        if pd.isna(row.get("posted_speed_limit"))
        else row["speed_mps"] > (row["posted_speed_limit"] + margin),
        axis=1,
    )

    # Aggregate
    agg = df.groupby("driver_id", observed=True).agg(
        trip_count=("trip_id", "nunique"),
        total_events=("event_id", "count"),
        avg_speed_mps=("speed_mps", "mean"),
        speed_std=("speed_mps", "std"),
        harsh_brake_rate=("is_harsh_brake", "mean"),
        harsh_accel_rate=("is_harsh_accel", "mean"),
        overspeed_rate=("is_overspeed", "mean"),
        night_trip_ratio=("is_night", "mean"),
        weekend_trip_ratio=("is_weekend", "mean"),
    )

    if total_km_by_driver is not None:
        agg = agg.merge(total_km_by_driver, on="driver_id", how="left")
        agg = agg.rename(columns={"trip_distance_km": "total_km"})
    else:
        agg["total_km"] = df.groupby("driver_id", observed=True)["odometer_km"].apply(
            lambda x: x.max() - x.min()
        )

    agg = agg.reset_index()

    # --- Enforce Gold schema contract ---
    schema_fields = load_gold_schema()
    for col in schema_fields:
        if col not in agg.columns:
            agg[col] = None
    agg = agg[schema_fields]

    # Write driver features
    out_file = os.path.join(GOLD_DIR, "driver_features.parquet")
    agg.to_parquet(
        out_file,
        index=False,
        engine="pyarrow",
        partition_cols=cfg["output"].get("partition_on", [])
    )
    logger.info("✅ Gold features written: %s (%s drivers)", out_file, len(agg))

    # Run manifest
    manifest = {
        "run_time": datetime.now(timezone.utc).isoformat(),
        "drivers": len(agg),
        "events_processed": len(df),
        "output_file": out_file,
    }
    manifest_file = cfg["output"].get("run_manifest", os.path.join(GOLD_DIR, "run_manifest.json"))
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Run manifest written: %s", manifest_file)


if __name__ == "__main__":
    build_features()
