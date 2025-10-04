#!/usr/bin/env python3
# === Phase 5: Training Set Validation ===
import os, json, pandas as pd, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(ROOT, "data/training/training_set.parquet")
MANIFEST_FILE = os.path.join(ROOT, "data/training/run_manifest.json")

if not os.path.exists(TRAIN_FILE):
    print(f"❌ Training set not found: {TRAIN_FILE}")
    sys.exit(1)
if not os.path.exists(MANIFEST_FILE):
    print(f"❌ Manifest not found: {MANIFEST_FILE}")
    sys.exit(1)

df = pd.read_parquet(TRAIN_FILE)
with open(MANIFEST_FILE) as f:
    manifest = json.load(f)

print("✅ Loaded training set and manifest")
print(f"- Rows in parquet: {len(df)}")
print(f"- Rows in manifest: {manifest.get('rows')}")
print(f"- Positives (claims): {df['claim_occurred'].sum()} (manifest: {manifest.get('positives', 'N/A')})")
print(f"- Mean severity: {df['claim_amount'].mean():.2f} (manifest: {manifest.get('mean_severity', 'N/A')})")

# Consistency checks
errors = []
if len(df) != manifest.get("rows"):
    errors.append("Row count mismatch")
if df['claim_occurred'].sum() != manifest.get("positives", df['claim_occurred'].sum()):
    errors.append("Positives mismatch")

if errors:
    print("❌ Validation failed:")
    for e in errors:
        print(" -", e)
    sys.exit(1)
else:
    print("✅ Validation passed: training set and manifest are consistent")
