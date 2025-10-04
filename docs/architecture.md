
## Phase A: Data Collection & Ingestion

### Medallion Flow
- **Raw** → Direct simulator output (Avro + Parquet) under `/data/raw` and `/data/sample`
- **Bronze/Silver/Gold** → To be populated in later phases (currently only Raw is available)

### Schema v4
- File: `/docs/schemas/telematics_event_v4.avsc`
- Covers:
  - **Core IDs & Time**: event_id, driver_id, trip_id, timestamps
  - **Telematics Core**: GPS, speed, acceleration, gyro, heading, odometer
  - **Driver Context**: age band, license years, history (accidents, claims, violations, policy tenure)
  - **Vehicle Context**: make, model year, safety rating, VIN hash, type/class, ownership
  - **Environment Context**: weather, road type, posted speed limit, crime index, accident index
- Data dictionary auto-generated at `/docs/data_dictionary_v4.md`

### Simulator
- Script: `/src/ingestion/simulator.py`
- Inputs:
  - `/config/simulator_full.yaml` → 50 drivers, 15–25 min trips
  - `/config/simulator_sample.yaml` → 5 drivers, 2 min trips
- Output:
  - Avro + Parquet batches written to `/data/raw/` (full) and `/data/sample/` (small demo)
- Logging includes batch write counts and total events.

### Validation
- Schema validated using `/bin/schema_check_v4.py`
- Data dictionary generated using `/bin/gen_data_dictionary.py`

## Phase 2: Collector API (Bronze Layer)

### Overview
The Collector API ingests telematics events from devices/apps, validates them against Schema v4, and persists to Bronze storage.
Invalid or malformed events are routed to a Dead Letter Queue (DLQ).

### Key Features
- **Schema Validation**: Avro schema v4 enforced at ingestion.
- **Extended Checks**: Lat/lon ranges, speed, accel, heading, age band, vehicle type, safety rating, indices.
- **Bronze Storage**: Valid events → `/data/bronze/` (Parquet batches).
- **DLQ**: Invalid events → `/data/dlq/events_dlq.jsonl`.
- **HMAC Authentication**: `X-Signature` header ensures trusted devices.
- **Idempotency**: LRU cache prevents duplicate events.
- **Rate Limiting**: Configurable per-driver via `/config/collector.yaml`.

### Endpoints
- `POST /ingest` → ingest one event.
- `GET /health` → metrics, bronze file count, DLQ size.
- `POST /flush` → flush in-memory buffers to disk.

### Configs
- Auth secrets → `/config/collector_auth.yaml`
- Rate limiting → `/config/collector.yaml`

### Raw → Bronze ETL (Batch Path)
In addition to the Collector API, the system supports a batch ETL pipeline:

- Script: `/src/ingestion/raw_to_bronze.py`
- Reads raw simulator outputs from `/data/raw/telematics_full/`
- Deduplicates on `event_id`
- Splits into chunks of configurable batch size
- Writes clean Bronze parquet files to `/data/bronze/`

This allows replay or backfill of historical simulator datasets into the Bronze layer.

## Phase 3: Bronze → Silver ETL

### Overview
The Bronze → Silver ETL pipeline cleanses, validates, and enriches raw Bronze events.
Silver data is schema-conformant, partitioned, and ready for downstream modeling.

### Key Steps
- **Deduplication**: Removes duplicate `event_id`s.
- **Sanity Filtering**: Enforces thresholds (lat/lon ranges, speed, accel, heading).
- **Partitioning**: Adds `event_date` and partitions by `driver_id`.
- **Schema Enforcement**: Ensures Silver schema contract (`telematics_event_v4_silver.avsc`).
- **Enrichment**: Derived trip distance, duration, average speed, harsh braking, harsh acceleration, overspeed events.

### Outputs
- Event-level Silver → `/data/silver/silver_events.parquet`
- Trip-level Silver → `/data/silver/silver_trips.parquet`

### Documentation
- Data dictionary available at: `docs/data_dictionary_silver.md`

### Sample Data (Phase 3)

To aid quick inspection, small sample files are provided:

- `data/sample/phase3/silver_events_sample.parquet` (20 events)
- `data/sample/phase3/silver_trips_sample.parquet` (5 trips)

These allow reviewers to validate schema conformance and enrichment without running the full ETL.

### Validation (Phase 3)

To ensure data integrity, a validation script is provided:

* **Path**: `bin/validate_silver.py`
* **Checks Performed**:
  - Confirms all fields from `telematics_event_v4_silver.avsc` are present.
  - Range validation: no negative speeds, lat ∈ [-90,90], lon ∈ [-180,180].
  - Enum validation: ensures valid values for fields like `source`.
  - Reports counts of violations if any.

This script allows quick data quality checks without re-running ETL.

### Profiling (Phase 3)

In addition to schema validation, automated profiling reports were generated to audit Silver data quality.

* **Event-level profiling**: `docs/profiling/phase3_events_report.html`
* **Trip-level profiling**: `docs/profiling/phase3_trips_report.html`

These reports provide:
- Field-level distributions and missing value percentages
- Descriptive statistics (mean, median, variance, quantiles)
- Cardinality and value counts for categorical fields
- Correlation heatmaps for numeric features
- Automated warnings (skewed distributions, constant columns, outliers)

This supports both data quality assurance and downstream feature engineering.

## Phase 4: Silver → Gold Feature Engineering

### Overview
The Silver → Gold pipeline generates **driver-level feature aggregates** used for risk scoring
and dynamic insurance pricing. These features capture driving style, trip behaviors, and contextual risk factors.

### Key Features
- **Trip statistics**: trip_count, total_events
- **Speed metrics**: avg_speed_mps, speed_std
- **Safety indicators**: harsh_brake_rate, harsh_accel_rate
- **Risk indicators**: overspeed_rate, night_trip_ratio, weekend_trip_ratio
- **Exposure metrics**: total_km traveled (from trips or odometer)

### Outputs
- Gold driver features → `/data/gold/driver_features.parquet` (partitioned by driver_id)
- Run manifest → `/data/gold/run_manifest.json`

### Purpose
Provides a structured, feature-rich Gold dataset for downstream ML models (risk scoring, pricing).

## Phase 4: Silver → Gold Feature Engineering

### Overview
The Silver → Gold pipeline generates **driver-level feature aggregates** used for risk scoring
and dynamic insurance pricing. These features capture driving style, trip behaviors, and contextual risk factors.

### Key Features
- **Trip statistics**: trip_count, total_events
- **Speed metrics**: avg_speed_mps, speed_std
- **Safety indicators**: harsh_brake_rate, harsh_accel_rate
- **Risk indicators**: overspeed_rate, night_trip_ratio, weekend_trip_ratio
- **Exposure metrics**: total_km traveled (from trips or odometer)

### Schema
Gold outputs conform to schema contract: `docs/schemas/driver_features_v1.avsc`.

### Outputs
- Gold driver features → `/data/gold/driver_features.parquet` (partitioned by driver_id)
- Run manifest → `/data/gold/run_manifest.json`
- Profiling report → `/docs/profiling/phase4_driver_features_report.html`

### Purpose
Provides a structured, feature-rich Gold dataset for downstream ML models (risk scoring, pricing).

## Phase 5: Gold → Training Set with Simulated Labels

### Overview
This phase transforms driver-level Gold features into a labeled training dataset.
Labels simulate **claim frequency** (binary outcome) and **claim severity** (gamma-distributed) with feature-driven weighting.

### Key Steps
- Apply logistic model with risk-weighted features (braking, acceleration, night ratio, trips).
- Simulate claim occurrence (`claim_occurred`) as binary frequency.
- Simulate claim severity (`claim_amount`) adjusted for exposure (km traveled) and volatility (speed variation).
- Enforce schema `training_labels_v1.avsc`.
- Generate a run manifest for reproducibility.

### Outputs
- Training set: `data/training/training_set.parquet`
- Run manifest: `data/training/run_manifest.json`

## Phase 6: Claim Prediction Models (Frequency & Severity)

### Overview
Phase 6 implements predictive modeling for **claim frequency** (likelihood of a claim within 30 days) 
and **claim severity** (expected cost conditional on a claim).  
This provides the foundation for **pure premium modeling** and dynamic UBI pricing.

### Frequency Models
- Logistic Regression (with scaling + calibration)
- Random Forest Classifier
- XGBoost Classifier (handles class imbalance via scale_pos_weight)

### Severity Models
- Linear Regression (baseline)
- XGBoost Regressor (non-linear effects, feature importance)
- Gamma GLM (distributionally appropriate for skewed positive costs)

### Enhancements
- Grouped train/test splits by `driver_id` (leakage prevention).
- Bootstrap confidence intervals for AUC/PR metrics.
- Model calibration with isotonic/sigmoid scaling.
- Model registry (`models/registry.json`) with versioning + metadata.
- Saved predictions, coefficients, and feature importances for interpretability.

### Outputs
- **Models:** `/models/frequency_*.pkl`, `/models/severity_*.pkl`
- **Predictions:** `/data/training/predictions_*.parquet`
- **Metrics:** `/data/training/*_metrics.json`
- **Registry:** `/models/registry.json`

### Purpose
Delivers risk prediction engines that directly support UBI pricing: 
frequency × severity = pure premium.

## Phase 6: Claim Prediction Models (Frequency & Severity)

### Overview
Phase 6 introduces supervised ML models that predict:
- **Frequency**: Probability of a driver filing a claim within 30 days.
- **Severity**: Expected cost of the claim (conditional on occurrence).

### Key Models
**Frequency (classification):**
- Logistic Regression (baseline, calibrated probabilities)
- Random Forest (tree ensemble, balanced class weights)
- XGBoost Classifier (gradient boosting with imbalance handling)

**Severity (regression, claimants only):**
- Linear Regression (interpretable baseline)
- XGBoost Regressor (nonlinear performance benchmark)
- Gamma GLM (distributionally appropriate severity model)

### Design
- **Data split**: Grouped by driver to avoid leakage across train/test.
- **Calibration**: Logistic/Platt calibration applied where feasible.
- **Evaluation**: Metrics reported with bootstrap confidence intervals.
- **Persistence**: Models stored under `/models/` and tracked in registry.
- **Outputs**: Predictions, metrics, and importances stored under `/data/training/`.

### Purpose
Provides production-ready models to simulate insurance risk pricing:
- Frequency models capture **likelihood of claims**.
- Severity models capture **magnitude of losses**.
- Together they enable actuarial-style **Pure Premium estimation**.


## Phase 6 Output: Driver Risk Scores

We combine the best **frequency** model (claim probability within 30 days) and the best **severity** model (expected claim size)
to compute **expected loss** per driver:

- `p_claim_30d`  = predicted probability from the top frequency model
- `severity_pred` = predicted claim severity from the top severity model
- `expected_loss` = `p_claim_30d * severity_pred`

Artifacts:
- Scores: `data/training/driver_risk_scores.parquet`
- Preview: `data/training/driver_risk_scores_preview.csv`
- Manifest: `data/training/driver_risk_scores_manifest.json`

Selection logic: the script picks the best models from `data/training/model_summary.csv` (highest AUC for frequency, lowest MAE for severity).
