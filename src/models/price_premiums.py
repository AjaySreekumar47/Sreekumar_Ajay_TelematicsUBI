# ============================================================================
# Pricing Engine: Convert risk scores into premiums
# ============================================================================
import os, pandas as pd, json
from datetime import datetime

REPO_NAME = "Sreekumar_Ajay_TelematicsUBI"
ROOT = f"/content/{REPO_NAME}"
DATA_DIR = f"{ROOT}/data/training"
INPUT_FILE = os.path.join(DATA_DIR, "driver_risk_scores.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "driver_premiums.parquet")
MANIFEST_FILE = os.path.join(DATA_DIR, "premium_manifest.json")

# Simple loadings (could be tuned actuarially)
EXPENSE_LOAD = 0.25   # 25% expense loading
MARGIN_LOAD = 0.10    # 10% target margin

def price_premiums():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Risk scores not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    # Expected loss already computed: p_claim_30d × severity_pred
    if "expected_loss" not in df.columns:
        raise KeyError("expected_loss missing from risk scores dataset")

    # Premium = expected loss × (1 + loadings)
    df["premium"] = df["expected_loss"] * (1 + EXPENSE_LOAD + MARGIN_LOAD)

    # Save results
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Premiums written: {OUTPUT_FILE} ({len(df)} rows)")

    manifest = {
        "input": INPUT_FILE,
        "output": OUTPUT_FILE,
        "rows": len(df),
        "expense_load": EXPENSE_LOAD,
        "margin_load": MARGIN_LOAD,
        "created_at": datetime.utcnow().isoformat()
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📝 Manifest saved: {MANIFEST_FILE}")

if __name__ == "__main__":
    price_premiums()
