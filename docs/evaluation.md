
## Phase 2: Collector API (Bronze Layer)

### Validation Evidence

* **Health check output (sample)**:

```json
{"status":"ok","metrics":{"accepted":0,"rejected":0,"unauth":0,"duplicates":0,"rate_limited":0},"bronze_files":0,"dlq_file_size":0}
```

* **Accepted event (SUV, uppercase)**:

```json
{"status":"accepted","queue_size":1}
```

* **Rejected event (suv, lowercase)**:

```json
{"status":"rejected","reason":"INVALID_VEHICLE_TYPE"}
```

* **DLQ contents (example)**:

```json
{"event_id":"...","driver_id":"...","vehicle_type":"suv","_dlq_reason":"INVALID_VEHICLE_TYPE"}
```


## Phase 3: Bronze → Silver ETL

### Validation Evidence

* **ETL logs (sample)**:

```
Rows — raw=50000, dedup=49980, filtered=49500
✅ Silver events written: data/silver/silver_events.parquet (49500 rows)
✅ Silver trips written: data/silver/silver_trips.parquet (50 rows)
```

* **Event-level preview**:

```
event_id, driver_id, trip_id, ts, event_date, lat, lon, speed_mps, accel_mps2, ...
```

* **Trip-level preview**:

```
driver_id, trip_id, trip_duration_sec, trip_distance_km, avg_speed_calc, harsh_brakes, harsh_accels, overspeed_events
```


## Phase 4: Silver → Gold Feature Engineering

### Validation Evidence

* **Sample run manifest**:

```
{
  "run_time": "2025-10-04T10:30:00Z",
  "drivers": 250,
  "events_processed": 50000,
  "output_file": "data/gold/driver_features.parquet"
}
```

* **Driver features preview**:

```
driver_id, trip_count, total_events, avg_speed_mps, speed_std,
harsh_brake_rate, harsh_accel_rate, overspeed_rate,
night_trip_ratio, weekend_trip_ratio, total_km
```

* **Data quality checks**:

- All rates ∈ [0,1]
- trip_count ≥ 0
- total_km non-negative

```

## Phase 4: Silver → Gold Feature Engineering

### Validation Evidence

* **Sample run manifest**:

```
{
  "run_time": "2025-10-04T10:30:00Z",
  "drivers": 250,
  "events_processed": 50000,
  "output_file": "data/gold/driver_features.parquet"
}
```

* **Driver features preview**:

```
driver_id, trip_count, total_events, avg_speed_mps, speed_std,
harsh_brake_rate, harsh_accel_rate, overspeed_rate,
night_trip_ratio, weekend_trip_ratio, total_km
```

* **Data quality checks**:

- All rates ∈ [0,1]
- trip_count ≥ 0
- total_km non-negative

* **Schema enforcement**:
  Validated against `docs/schemas/driver_features_v1.avsc`.

* **Profiling report**:
  See `docs/profiling/phase4_driver_features_report.html` for feature distributions, correlations, and quality metrics.

## Phase 5: Training Set Generation

### Validation Evidence

* **Manifest Example**

```
{
  "rows": 200,
  "seed": 42,
  "base_claim_probability": 0.2,
  "gamma_shape": 2.0,
  "gamma_scale": 500.0,
  "feature_weighting": true,
  "run_date": "2025-10-04T12:45:00Z"
}
```

* **Label Distribution (claim_occurred)**

- 1 → ~20%
- 0 → ~80%

* **Claim Severity Stats**

- mean: ~700
- max: ~3000
- non-claims: 0.0

```

### Validation Script

A built-in validation script is provided to cross-check consistency between the
generated training set and its manifest.

```bash
cd Sreekumar_Ajay_TelematicsUBI
bin/validate_training.sh
````

**Checks performed:**

* Row count matches between `training_set.parquet` and manifest
* Positive claim counts (`claim_occurred`) align
* Mean severity close to manifest stats
* Exits with non-zero code if validation fails

This ensures reproducibility and prevents silent data drift in training set generation.

### Reporting

A reporting script is provided to generate a quick summary of the
training set for evaluation purposes:

```bash
cd Sreekumar_Ajay_TelematicsUBI
bin/report_training.sh
````

**Outputs include:**

* Label distribution (`claim_occurred`)
* Claim severity statistics (non-zero claims)
* Key feature averages (e.g., speed, harsh braking, overspeed, night driving)
* Row and column counts

This helps reviewers quickly assess realism and quality of the generated dataset.

## Phase 6: Claim Prediction Models

### Validation Evidence

* **Frequency model metrics (sample)**:

```
Logistic Regression — AUC: 0.64, PR: 0.28, Brier: 0.178
Random Forest       — AUC: 0.61, PR: 0.25, Brier: 0.182
XGBoost Classifier  — AUC: 0.66, PR: 0.31, Brier: 0.176
```

* **Severity model metrics (sample)**:

```
Linear Regression — MAE: $620.15, RMSE: $1050.22, R²: 0.12
XGBoost Regressor — MAE: $580.45, RMSE: $990.33,  R²: 0.20
Gamma GLM         — MAE: $600.72, RMSE: $1015.11, R²: 0.18
```

* **Artifacts generated**:

- Predictions in `/data/training/predictions_*.parquet`
- Metrics in `/data/training/*_metrics.json`
- Feature importances / coefficients parquet files
- Saved models in `/models/*.pkl`
- Registry entry appended to `/models/registry.json`

## Phase 6: Model Performance Summary

| Model | AUC | PR | Brier | MAE | RMSE | R² | Run Time | Config Hash |
|-------|-----|----|-------|-----|------|----|----------|-------------|
| Frequency — Logistic Regression | 0.38 | 0.235 | 0.219 | - | - | - | - | - |
| Frequency — Random Forest | 0.64 | 0.346 | 0.183 | - | - | - | - | - |
| Frequency — XGBoost Classifier | 0.613 | 0.424 | 0.209 | - | - | - | - | - |
| Severity — Linear Regression | - | - | - | 1758.37 | 2307.88 | -12.796 | - | - |
| Severity — XGBoost Regressor | - | - | - | 716.44 | 832.89 | -0.797 | - | - |
| Severity — Gamma GLM | - | - | - | 1277.33 | 1414.44 | -4.182 | - | - |

## Phase 6: Claim Prediction Models

### Validation Evidence

* **Summary of Model Performance (actual run):**

| Model                           | AUC   | PR    | Brier | MAE    | RMSE    | R²     |
| ------------------------------- | ----- | ----- | ----- | ------ | ------- | ------ |
| Frequency — Logistic Regression | 0.531 | 0.248 | 0.160 | –      | –       | –      |
| Frequency — Random Forest       | 0.578 | 0.407 | 0.180 | –      | –       | –      |
| Frequency — XGBoost Classifier  | 0.453 | 0.214 | 0.233 | –      | –       | –      |
| Severity — Linear Regression    | –     | –     | –     | 731.50 | 864.10  | -0.026 |
| Severity — XGBoost Regressor    | –     | –     | –     | 499.09 | 605.82  | 0.496  |
| Severity — Gamma GLM            | –     | –     | –     | 951.00 | 1019.81 | -0.429 |

* **Artifacts:**

- Predictions parquet files stored in `/data/training/`.
- Metrics JSONs per model (AUC, PR, MAE, RMSE, etc.).
- Registry file tracks model versions and hashes.

* **Checks:**

- Frequency labels balanced across train/test.
- Severity models only trained on claimants (non-zero).
- Outputs reproducible via config + registry hash.
