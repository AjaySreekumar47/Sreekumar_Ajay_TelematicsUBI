#!/usr/bin/env python3
import os, json, pandas as pd, sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SILVER_DIR = os.path.join(ROOT, "data", "silver")
SCHEMA_PATH = os.path.join(ROOT, "docs", "schemas", "telematics_event_v4_silver.avsc")

def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)

def validate():
    schema = load_schema()
    fields = [f["name"] for f in schema["fields"]]

    # Load event parquet files
    event_files = []
    for root, dirs, files in os.walk(SILVER_DIR):
        for f in files:
            if f.endswith(".parquet") and ("silver_events" in f or "event_date=" in root):
                event_files.append(os.path.join(root, f))

    if not event_files:
        print("❌ No Silver event files found")
        sys.exit(1)

    df = pd.concat([pd.read_parquet(f) for f in event_files], ignore_index=True)

    # 1. Schema check
    missing = [c for c in fields if c not in df.columns]
    if missing:
        print("❌ Missing columns:", missing)
    else:
        print("✅ All schema fields present")

    # 2. Range checks
    bad_speeds = df[df["speed_mps"] < 0]
    bad_lat = df[(df["lat"] < -90) | (df["lat"] > 90)]
    bad_lon = df[(df["lon"] < -180) | (df["lon"] > 180)]

    print(f"Speed violations: {len(bad_speeds)}")
    print(f"Lat violations: {len(bad_lat)}")
    print(f"Lon violations: {len(bad_lon)}")

    # 3. Enum checks
    if "source" in df.columns:
        valid_sources = {"phone","OBD","simulator"}
        bad_sources = set(df["source"].dropna().unique()) - valid_sources
        if bad_sources:
            print("❌ Invalid sources:", bad_sources)
        else:
            print("✅ All sources valid")

    print(f"✅ Validation complete on {len(df)} rows")

if __name__ == "__main__":
    validate()
