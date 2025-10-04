#!/usr/bin/env python3
"""
Generate a Markdown Data Dictionary from Avro schema (polished version).

Enhancements:
- Groups fields into sections (Core, Telematics, Driver, Vehicle, Environment).
- Expands enum symbols inline for readability.
- Works with any TelematicsEvent schema version (v2–v4).
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Union

def setup_logger(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

def normalize_type(t: Union[str, dict, list]) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        if t.get("type") == "enum":
            return f"enum({','.join(t.get('symbols', []))})"
        return t.get("type", str(t))
    if isinstance(t, list):
        return "|".join([normalize_type(x) for x in t])
    return str(t)

# Section mapping based on field names
SECTION_MAP = {
    "Core IDs & Time": {"event_id","driver_id","trip_id","ts","ingestion_ts","event_seq_no"},
    "Telematics Core": {"lat","lon","speed_mps","accel_mps2","gyro","heading","odometer_km","source"},
    "Driver Context": {"driver_age_band","license_years","accidents_count","claims_count","violation_points","policy_tenure_years"},
    "Vehicle Context": {"vehicle_make","vehicle_model_year","safety_rating","vehicle_vin_hash","vehicle_type","ownership_type"},
    "Environment Context": {"weather","road_type","posted_speed_limit","crime_index","accident_index"},
}

def generate_markdown(schema: Dict[str, Any]) -> str:
    fields = schema.get("fields", [])
    # Map field name -> spec
    field_map = {f["name"]: f for f in fields}

    lines = []
    lines.append("# 📑 Data Dictionary\n")

    for section, fnames in SECTION_MAP.items():
        present = [field_map[f] for f in fnames if f in field_map]
        if not present:
            continue
        lines.append(f"\n## {section}\n")
        lines.append("| Field | Type | Description |")
        lines.append("|-------|------|-------------|")
        for f in present:
            name = f["name"]
            ftype = normalize_type(f["type"])
            desc = f.get("doc", "").replace("\n"," ")
            lines.append(f"| {name} | {ftype} | {desc} |")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate Data Dictionary from Avro schema")
    parser.add_argument("--schema", required=True, help="Path to schema .avsc")
    parser.add_argument("--out", required=True, help="Output .md file path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logger(args.log_level)

    try:
        with open(args.schema) as f:
            schema = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read schema: {e}")
        sys.exit(1)

    md = generate_markdown(schema)
    try:
        with open(args.out, "w") as f:
            f.write(md)
    except Exception as e:
        logging.error(f"Failed to write data dictionary: {e}")
        sys.exit(1)

    logging.info("✅ Data dictionary written to %s", args.out)

if __name__ == "__main__":
    main()
