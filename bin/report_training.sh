#!/usr/bin/env python3
# === Phase 5: Training Set Reporting ===
import os, pandas as pd, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(ROOT, "data/training/training_set.parquet")

if not os.path.exists(TRAIN_FILE):
    print(f"❌ Training set not found: {TRAIN_FILE}")
    sys.exit(1)

df = pd.read_parquet(TRAIN_FILE)

print("📊 Training Set Report")
print("="*60)
print(f"Rows (drivers): {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Label distribution
print("Label distribution (claim_occurred):")
print(df['claim_occurred'].value_counts(normalize=True).round(3))
print()

# Severity stats
print("Claim severity statistics (non-zero claims):")
claims = df.loc[df['claim_occurred'] == 1, 'claim_amount']
if len(claims) > 0:
    print(claims.describe().round(2))
else:
    print("No claims in this dataset.")
print()

# Feature stats (sample subset)
features = ["trip_count", "avg_speed_mps", "harsh_brake_rate", "harsh_accel_rate", "overspeed_rate", "night_trip_ratio", "total_km"]
available = [f for f in features if f in df.columns]
print("Key feature averages:")
print(df[available].mean().round(3).to_string())
