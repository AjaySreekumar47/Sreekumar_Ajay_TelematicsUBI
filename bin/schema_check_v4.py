"""
Schema Validator v4 for TelematicsEvent (Production-ready)

Extends v2/v3 by validating:
- Core telemetry fields (same as before)
- Driver history (accidents, claims, violations, policy tenure)
- Extended vehicle info (VIN hashed, type/class, ownership)
- Environmental indices (crime_index, accident_index)
"""

import argparse, json, logging, sys
from typing import Any, Dict, List, Union

# === Required fields (same as v2) ===
REQUIRED_FIELDS = {
    "event_id": "string",
    "driver_id": "string",
    "trip_id": "string",
    "ts": "long",                # must carry logicalType
    "ingestion_ts": "long",      # must carry logicalType
    "event_seq_no": "int",
    "lat": "double",
    "lon": "double",
    "speed_mps": "double",
    "odometer_km": "double",
    "source": "enum",
}

# === Optional fields (now includes context + history) ===
OPTIONAL_FIELDS = {
    "accel_mps2": ["null", "double"],
    "gyro": ["null", "double"],
    "heading": ["null", "double"],

    # Driver context
    "driver_age_band": ["null", "enum"],
    "license_years": ["null", "int"],
    "accidents_count": ["null", "int"],
    "claims_count": ["null", "int"],
    "violation_points": ["null", "int"],
    "policy_tenure_years": ["null", "int"],

    # Vehicle context
    "vehicle_make": ["null", "string"],
    "vehicle_model_year": ["null", "int"],
    "safety_rating": ["null", "float"],
    "vehicle_vin_hash": ["null", "string"],
    "vehicle_type": ["null", "enum"],
    "ownership_type": ["null", "enum"],

    # Environment context
    "weather": ["null", "enum"],
    "road_type": ["null", "enum"],
    "posted_speed_limit": ["null", "int"],
    "crime_index": ["null", "float"],
    "accident_index": ["null", "float"],
}

EXPECTED_ENUMS = {
    "source": ["phone", "OBD", "simulator"],
    "driver_age_band": ["UNDER25","AGE_25_40","AGE_40_60","OVER60"],
    "vehicle_type": ["SEDAN","SUV","TRUCK","MOTORCYCLE","EV","OTHER"],
    "ownership_type": ["OWNED","LEASED","FLEET"],
    "weather": ["clear","rain","snow","fog","other"],
    "road_type": ["highway","urban","rural"],
}

def setup_logger(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

def normalize_type(t: Union[str, dict, list]) -> Union[str, List[str]]:
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        return t.get("type")
    if isinstance(t, list):
        return sorted([normalize_type(x) if isinstance(x, dict) else x for x in t])
    return str(t)

def validate_schema(schema: Dict[str, Any]) -> bool:
    ok = True
    fields = {f["name"]: f for f in schema.get("fields", [])}

    # 1) Required presence + type
    for fname, ftype in REQUIRED_FIELDS.items():
        if fname not in fields:
            logging.error(f"Missing required field: {fname}")
            ok = False
            continue
        f = fields[fname]
        normalized = normalize_type(f["type"])
        if ftype == "enum":
            if not isinstance(f["type"], dict) or f["type"].get("type") != "enum":
                logging.error(f"{fname} must be enum, found {f['type']}")
                ok = False
        elif normalized != ftype:
            logging.error(f"{fname} must be {ftype}, found {normalized}")
            ok = False

    # 2) Optional unions + default:null
    for fname, expected_types in OPTIONAL_FIELDS.items():
        if fname not in fields:
            logging.error(f"Missing optional field: {fname}")
            ok = False
            continue
        f = fields[fname]
        normalized = normalize_type(f["type"])
        if normalized != sorted(expected_types):
            logging.error(f"{fname} must allow {expected_types}, found {normalized}")
            ok = False
        if f.get("default", None) is not None:
            logging.error(f"{fname} must have default=null, found {f.get('default')}")
            ok = False

    # 3) Enum symbols checks
    for fname, expected_syms in EXPECTED_ENUMS.items():
        f = fields.get(fname)
        if f and isinstance(f["type"], dict) and f["type"].get("type") == "enum":
            symbols = f["type"].get("symbols", [])
            if sorted(symbols) != sorted(expected_syms):
                logging.error(f"{fname} enum symbols mismatch. Expected {expected_syms}, found {symbols}")
                ok = False

    # 4) Logical constraints for timestamps
    for ts_field in ["ts", "ingestion_ts"]:
        f = fields.get(ts_field)
        if not isinstance(f["type"], dict) or f["type"].get("logicalType") != "timestamp-millis":
            logging.error(f"{ts_field} must have logicalType=timestamp-millis")
            ok = False

    # 5) Docstrings completeness
    for fname, f in fields.items():
        if "doc" not in f:
            logging.warning(f"{fname} missing doc string")

    return ok

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schema", required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    setup_logger(args.log_level)
    try:
        with open(args.schema) as f:
            schema = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load schema: {e}")
        sys.exit(1)

    if validate_schema(schema):
        logging.info("✅ Schema v4 validation passed")
        sys.exit(0)
    else:
        logging.error("❌ Schema v4 validation failed")
        sys.exit(2)

if __name__ == "__main__":
    main()
