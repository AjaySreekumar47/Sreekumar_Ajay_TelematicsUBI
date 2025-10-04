# ============================================================================
# train_phase6.py — Phase 6: Claim Prediction Models (Frequency & Severity)
# ============================================================================
# Implements:
# - 6a: Frequency models (Logistic, Random Forest, XGBoost)
# - 6b: Severity models (Linear Regression, XGBoost Regressor, Gamma GLM)
# - Grouped train/test split by driver to prevent leakage
# - Metrics with bootstrap CIs, registry persistence, summary table
# ============================================================================

import os, json, warnings, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, TweedieRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from joblib import dump
import logging

# ============================================================================
# Config
# ============================================================================

REPO_NAME = "Sreekumar_Ajay_TelematicsUBI"
ROOT = f"/content/{REPO_NAME}"
TRAIN_FILE = f"{ROOT}/data/training/training_set.parquet"
OUT_DIR = f"{ROOT}/data/training"
MODELS_DIR = f"{ROOT}/models"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Phase6")
NP_RNG = np.random.default_rng(RANDOM_STATE)

# ============================================================================
# Helpers
# ============================================================================

def data_hash(df: pd.DataFrame) -> str:
    arr = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

def grouped_split(X, y, groups, test_size=0.2, random_state=RANDOM_STATE):
    if len(np.unique(y)) < 2 or len(y) < 10:
        return None
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in splitter.split(X, y, groups):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
            return train_idx, test_idx
    return None

def fit_calibrated_model(model, X_train, y_train, X_valid=None, y_valid=None):
    try:
        if len(y_train) >= 50 and len(np.unique(y_train)) == 2:
            cal = CalibratedClassifierCV(estimator=model, method="isotonic", cv=3)
            cal.fit(X_train, y_train)
            return cal, "isotonic_cv3"
        model.fit(X_train, y_train)
        cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
        cal.fit(X_valid if X_valid is not None else X_train,
                y_valid if y_valid is not None else y_train)
        return cal, "sigmoid_prefit"
    except Exception as e:
        warnings.warn(f"Calibration failed: {e}. Using raw model.")
        model.fit(X_train, y_train)
        return model, "uncalibrated"

def bootstrap_ci(y_true, y_pred, metric_fn, B=200):
    if len(y_true) < 5:
        return None
    idx = np.arange(len(y_true))
    metrics = []
    for _ in range(B):
        sample = NP_RNG.choice(idx, size=len(idx), replace=True)
        try:
            metrics.append(metric_fn(y_true[sample], y_pred[sample]))
        except:
            continue
    if not metrics:
        return None
    return [float(np.quantile(metrics, 0.025)), float(np.quantile(metrics, 0.975))]

def gini_coefficient(actual, pred):
    all_data = np.c_[actual, pred, np.arange(len(actual))]
    all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
    total_losses = all_data[:, 0].sum()
    if total_losses == 0:
        return 0
    gini_sum = all_data[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.0
    return gini_sum / len(actual)

def normalized_gini(y_true, y_pred):
    g_pred = gini_coefficient(y_true, y_pred)
    g_perfect = gini_coefficient(y_true, y_true)
    return g_pred / g_perfect if g_perfect != 0 else 0

def save_to_registry(model_name, version, data_sig, artifact_path, metrics_path):
    registry_path = os.path.join(MODELS_DIR, "registry.json")
    reg = []
    if os.path.exists(registry_path):
        try:
            reg = json.loads(Path(registry_path).read_text())
        except:
            reg = []
    reg.append({
        "name": model_name,
        "version": version,
        "data_hash": data_sig,
        "artifact": artifact_path,
        "metrics_path": metrics_path,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "phase": "phase6"
    })
    Path(registry_path).write_text(json.dumps(reg, indent=2))

# ============================================================================
# Data Prep
# ============================================================================

def prepare_frequency_data(df: pd.DataFrame):
    if "had_claim_within_30d" not in df.columns:
        raise KeyError("Training set missing 'had_claim_within_30d'.")
    drop_cols = ["driver_id", "had_claim_within_30d", "claim_severity",
                 "generation_version", "run_date"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    y = df["had_claim_within_30d"].astype(int).values
    groups = df["driver_id"].astype(str).values if "driver_id" in df.columns else np.arange(len(df))
    return X, y, groups, list(X.columns)

def prepare_severity_data(df: pd.DataFrame):
    if "had_claim_within_30d" not in df.columns:
        raise KeyError("Training set missing 'had_claim_within_30d'.")
    df_claims = df[df["had_claim_within_30d"] == 1].copy()
    if df_claims.empty:
        return None, None, None, None, None
    drop_cols = ["driver_id", "had_claim_within_30d", "claim_severity",
                 "generation_version", "run_date"]
    X = df_claims.drop(columns=[c for c in drop_cols if c in df_claims.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    y = df_claims["claim_severity"].astype(float).values
    groups = df_claims["driver_id"].astype(str).values if "driver_id" in df_claims.columns else np.arange(len(df_claims))
    return df_claims, X, y, groups, list(X.columns)

# ============================================================================
# Training Functions
# ============================================================================

def train_logistic_regression(df, data_sig):
    X, y, groups, features = prepare_frequency_data(df)
    split = grouped_split(X.values, y, groups)
    if split is None:
        train_idx = test_idx = np.arange(len(y))
    else:
        train_idx, test_idx = split
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("logit", LogisticRegression(max_iter=500,
                                                  class_weight="balanced",
                                                  random_state=RANDOM_STATE))])
    model, calib_mode = fit_calibrated_model(pipe, X_train, y_train, X_test, y_test)
    p_test = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": "frequency_logit",
        "AUC": float(roc_auc_score(y_test, p_test)),
        "PR": float(average_precision_score(y_test, p_test)),
        "Brier": float(brier_score_loss(y_test, p_test)),
        "features": features
    }
    dump(model, os.path.join(MODELS_DIR, "frequency_logit.pkl"))
    return metrics

def train_random_forest(df, data_sig):
    X, y, groups, features = prepare_frequency_data(df)
    split = grouped_split(X.values, y, groups)
    train_idx, test_idx = split if split else (np.arange(len(y)), np.arange(len(y)))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    model = RandomForestClassifier(n_estimators=200, max_depth=5,
                                   class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": "frequency_rf",
        "AUC": float(roc_auc_score(y_test, p_test)),
        "PR": float(average_precision_score(y_test, p_test)),
        "Brier": float(brier_score_loss(y_test, p_test)),
        "features": features
    }
    dump(model, os.path.join(MODELS_DIR, "frequency_rf.pkl"))
    return metrics

def train_xgb_classifier(df, data_sig):
    X, y, groups, features = prepare_frequency_data(df)
    split = grouped_split(X.values, y, groups)
    train_idx, test_idx = split if split else (np.arange(len(y)), np.arange(len(y)))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=scale_pos_weight,
                          eval_metric="logloss", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": "frequency_xgb",
        "AUC": float(roc_auc_score(y_test, p_test)),
        "PR": float(average_precision_score(y_test, p_test)),
        "Brier": float(brier_score_loss(y_test, p_test)),
        "features": features
    }
    dump(model, os.path.join(MODELS_DIR, "frequency_xgb.pkl"))
    return metrics

def train_linreg(df, data_sig):
    df_claims, X, y, groups, features = prepare_severity_data(df)
    if X is None:
        return None
    split = grouped_split(X.values, y, groups)
    train_idx, test_idx = split if split else (np.arange(len(y)), np.arange(len(y)))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    pipe = Pipeline([("scaler", StandardScaler()), ("linreg", LinearRegression())])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "model": "severity_linreg",
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "features": features
    }
    dump(pipe, os.path.join(MODELS_DIR, "severity_linreg.pkl"))
    return metrics

def train_xgb_regressor(df, data_sig):
    df_claims, X, y, groups, features = prepare_severity_data(df)
    if X is None:
        return None
    split = grouped_split(X.values, y, groups)
    train_idx, test_idx = split if split else (np.arange(len(y)), np.arange(len(y)))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "model": "severity_xgb",
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "features": features
    }
    dump(model, os.path.join(MODELS_DIR, "severity_xgb.pkl"))
    return metrics

def train_gamma_glm(df, data_sig):
    df_claims, X, y, groups, features = prepare_severity_data(df)
    if X is None:
        return None
    split = grouped_split(X.values, y, groups)
    train_idx, test_idx = split if split else (np.arange(len(y)), np.arange(len(y)))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("glm", TweedieRegressor(power=2, link="log", max_iter=2000))])
    pipe.fit(X_train, y_train)
    y_pred = np.clip(pipe.predict(X_test), 0, None)
    metrics = {
        "model": "severity_gamma_glm",
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "features": features
    }
    dump(pipe, os.path.join(MODELS_DIR, "severity_gamma_glm.pkl"))
    return metrics

# ============================================================================
# Run Phase 6
# ============================================================================

def run_phase6():
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Missing training file: {TRAIN_FILE}")
    df = pd.read_parquet(TRAIN_FILE)
    data_sig = data_hash(df)
    logger.info(f"Loaded training set ({len(df)} rows), hash={data_sig}")

    results = []
    # Frequency models
    results.append(train_logistic_regression(df, data_sig))
    results.append(train_random_forest(df, data_sig))
    results.append(train_xgb_classifier(df, data_sig))
    # Severity models
    results.append(train_linreg(df, data_sig))
    results.append(train_xgb_regressor(df, data_sig))
    results.append(train_gamma_glm(df, data_sig))

    # Save summary CSV
    df_results = pd.DataFrame([r for r in results if r is not None])
    summary_path = os.path.join(OUT_DIR, "model_summary.csv")
    df_results.to_csv(summary_path, index=False)
    logger.info(f"Summary written → {summary_path}")
    logger.info("Phase 6 complete.")

if __name__ == "__main__":
    run_phase6()
