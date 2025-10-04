
# === Fixed: Produce driver-level risk scores ===
import os, json, numpy as np, pandas as pd, joblib
from pathlib import Path
from datetime import datetime

ROOT = "/content/Sreekumar_Ajay_TelematicsUBI"
TRAIN_FILE = f"{ROOT}/data/training/training_set.parquet"
SUMMARY_CSV = f"{ROOT}/data/training/model_summary.csv"
OUT_PARQUET = f"{ROOT}/data/training/driver_risk_scores.parquet"
OUT_PREVIEW = f"{ROOT}/data/training/driver_risk_scores_preview.csv"
OUT_MANIFEST = f"{ROOT}/data/training/driver_risk_scores_manifest.json"

def _best_models(summary_csv):
    if not os.path.exists(summary_csv):
        return f"{ROOT}/models/frequency_rf.pkl", f"{ROOT}/models/severity_xgb.pkl"
    df = pd.read_csv(summary_csv)
    freq = df[df["Model"].str.contains("Frequency", na=False)]
    sev = df[df["Model"].str.contains("Severity", na=False)]
    freq_art = freq.sort_values("AUC", ascending=False).iloc[0]["artifact"] if not freq.empty else f"{ROOT}/models/frequency_rf.pkl"
    sev_art = sev.sort_values("MAE", ascending=True).iloc[0]["artifact"] if not sev.empty else f"{ROOT}/models/severity_xgb.pkl"
    return freq_art, sev_art

def _prepare_X(df, target_cols):
    drop_cols = ["driver_id","generation_version","run_date"] + target_cols
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").select_dtypes(include=[np.number])

def _predict_proba_safely(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    else:
        return model.predict(X).astype(float)

def _predict_positive_safely(model, X):
    return np.clip(model.predict(X), 0, None)

def main():
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Missing training file {TRAIN_FILE}")
    df = pd.read_parquet(TRAIN_FILE)

    # 🔧 Normalize column names if old ones exist
    rename_map = {
        "claim_occurred": "had_claim_within_30d",
        "claim_amount": "claim_severity"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "had_claim_within_30d" not in df.columns or "claim_severity" not in df.columns:
        raise KeyError("Training set missing expected claim columns after normalization.")

    freq_art, sev_art = _best_models(SUMMARY_CSV)
    freq_model, sev_model = joblib.load(freq_art), joblib.load(sev_art)

    X_freq = _prepare_X(df, ["had_claim_within_30d","claim_severity"])
    X_sev  = _prepare_X(df, ["had_claim_within_30d","claim_severity"])

    p_claim = np.clip(_predict_proba_safely(freq_model, X_freq), 0, 1)
    sev_pred = _predict_positive_safely(sev_model, X_sev)
    exp_loss = p_claim * sev_pred

    out = pd.DataFrame({
        "driver_id": df["driver_id"].values,
        "p_claim_30d": p_claim,
        "severity_pred": sev_pred,
        "expected_loss": exp_loss,
    })
    if "total_km" in df.columns: out["total_km"] = df["total_km"]

    out.to_parquet(OUT_PARQUET, index=False)
    out.head(20).to_csv(OUT_PREVIEW, index=False)

    manifest = {
        "run_time": datetime.utcnow().isoformat(),
        "rows": len(out),
        "drivers": int(out["driver_id"].nunique()),
        "frequency_model": os.path.relpath(freq_art, ROOT),
        "severity_model": os.path.relpath(sev_art, ROOT),
        "output_parquet": os.path.relpath(OUT_PARQUET, ROOT)
    }
    Path(OUT_MANIFEST).write_text(json.dumps(manifest, indent=2))
    print("✅ Risk scores written:", OUT_PARQUET)

if __name__ == "__main__":
    main()
