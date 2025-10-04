
# === Phase 5 : Label generation with feature-driven risk + manifest (Final) ===
import os, pandas as pd, numpy as np, yaml, logging, json, hashlib
from datetime import datetime

REPO_NAME = "Sreekumar_Ajay_TelematicsUBI"
ROOT = f"/content/{REPO_NAME}"

CFG_PATH = f"{ROOT}/config/labels.yaml"
GOLD_FILE = f"{ROOT}/data/gold/driver_features.parquet"
TRAIN_DIR = f"{ROOT}/data/training"
SCHEMA_PATH = f"{ROOT}/docs/schemas/training_labels_v1.avsc"

os.makedirs(TRAIN_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("build_labels")

def load_cfg(path=CFG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def hash_config(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def enforce_schema(df: pd.DataFrame, schema_path: str) -> pd.DataFrame:
    with open(schema_path) as f:
        schema = json.load(f)
    schema_fields = [f["name"] for f in schema["fields"]]
    for col in schema_fields:
        if col not in df.columns:
            df[col] = None
    return df[schema_fields]

def build_labels():
    cfg = load_cfg()["labels"]

    if not os.path.exists(GOLD_FILE):
        logger.error("Gold features not found: %s", GOLD_FILE)
        return

    df = pd.read_parquet(GOLD_FILE)
    logger.info("Loaded %s Gold feature rows", len(df))

    rng = np.random.default_rng(cfg.get("random_seed", 42))

    # --- Base probability
    base_p = cfg["claim_probability"]

    # --- Weighted risk score from features
    if cfg.get("use_feature_weights", False):
        risk_score = (
            0.6 * df["harsh_brake_rate"].fillna(0) +
            0.6 * df["harsh_accel_rate"].fillna(0) +
            0.2 * df["night_trip_ratio"].fillna(0) +
            0.001 * df["trip_count"].fillna(0)
        )
        # logistic scaling: keeps probs in (0,1)
        probs = logistic(np.log(base_p/(1-base_p)) + risk_score)
    else:
        probs = np.full(len(df), base_p)

    # --- Frequency label
    claim_occurred = rng.binomial(1, probs)

    # --- Severity label
    sev_cfg = cfg["claim_severity"]
    severities = rng.gamma(sev_cfg["shape"], sev_cfg["scale"], size=len(df))

    # scale severity by exposure (total_km) and volatility (speed_std)
    exposure_factor = (1 + df["total_km"].fillna(0) / 1000)
    volatility_factor = (1 + df["speed_std"].fillna(0) / 10)
    claim_amount = claim_occurred * severities * exposure_factor * volatility_factor

    # --- Assemble output
    df_out = df.copy()
    df_out["claim_occurred"] = claim_occurred.astype(bool)
    df_out["claim_amount"] = claim_amount
    df_out["claim_probability_sim"] = probs
    df_out["generation_version"] = "v5"
    df_out["run_date"] = datetime.utcnow().isoformat()

    # --- Enforce schema
    df_out = enforce_schema(df_out, SCHEMA_PATH)

    # --- Write training set
    out_file = os.path.join(TRAIN_DIR, "training_set.parquet")
    df_out.to_parquet(out_file, index=False, engine="pyarrow")
    logger.info("✅ Training set written: %s (%s rows)", out_file, len(df_out))

    # --- Manifest logging
    manifest = {
        "run_time": datetime.utcnow().isoformat(),
        "config_hash": hash_config(cfg),
        "rows": len(df_out),
        "positives": int(df_out["claim_occurred"].sum()),
        "claim_rate": float(df_out["claim_occurred"].mean()),
        "mean_severity": float(df_out["claim_amount"].mean()),
        "max_severity": float(df_out["claim_amount"].max()),
        "gamma_shape": sev_cfg["shape"],
        "gamma_scale": sev_cfg["scale"],
        "base_claim_probability": base_p,
        "feature_weighting": cfg.get("use_feature_weights", False),
    }
    manifest_file = os.path.join(TRAIN_DIR, "run_manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written: %s", manifest_file)

if __name__ == "__main__":
    build_labels()
