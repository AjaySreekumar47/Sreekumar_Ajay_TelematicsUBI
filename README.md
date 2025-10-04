# 🚗 Telematics Usage-Based Insurance (UBI) POC

This project implements a **telematics-driven insurance pricing engine** that leverages driver behavior data (speed, braking, acceleration, mileage, trip times) to dynamically calculate fairer insurance premiums. It demonstrates an end-to-end pipeline from raw telematics ingestion to machine learning risk modeling and a production-ready monitoring dashboard.

---

## 📌 Objectives
- Improve **premium accuracy** using real-world driving behavior.
- Encourage **safer driving habits** through usage-based incentives.
- Provide **transparent risk scoring** to customers.
- Demonstrate compliance-conscious, scalable data processing.

---

## 📂 Repository Structure
````

Sreekumar_Ajay_TelematicsUBI/
│
├── /src/          # Source code (ingestion, feature engineering, modeling, pricing engine, dashboard)
├── /data/         # Sample simulated data (bronze/silver/gold/training)
├── /models/       # Trained models + registry
├── /bin/          # Executable scripts (run ingestion, training, dashboard, pricing)
├── /docs/         # Design docs, architecture diagrams, phase notes
├── /docker/       # Dockerfile + entrypoint
└── README.md      # This file (setup & usage instructions)

````

---

## ⚙️ Setup Instructions

Docker (Self-contained)
```bash
docker build -t telematics-ubi .
docker run -p 8501:8501 telematics-ubi
````

This launches the Streamlit dashboard at [http://localhost:8501](http://localhost:8501).

---

## 🚀 Run Instructions

### 1. Data Ingestion

```bash
bash bin/run_ingestion.sh
```

### 2. Feature Engineering

```bash
bash bin/run_features.sh
```

### 3. Label Generation

```bash
bash bin/run_labels.sh
```

### 4. Train Models (Frequency & Severity)

```bash
bash bin/run_training.sh
```

### 5. Produce Risk Scores

```bash
bash bin/run_produce_risk_scores.sh
```

### 6. Pricing Engine

```bash
bash bin/run_pricing.sh
```

### 7. Dashboard

```bash
bash bin/run_dashboard.sh
```

---

## 📊 Evaluation

* **Model Metrics** (saved in `/data/training` as JSON + parquet):

  * Frequency: AUC, PR, Brier.
  * Severity: MAE, RMSE, R², NormGini.
* **Risk Scores**: driver_risk_scores.parquet.
* **Premium Outputs**: premium_quotes.parquet.
* **Dashboard**: Interactive monitoring via Streamlit (Phase 7).

---

## 📝 Notes

* Data used is **synthetic telematics POC data** (no sensitive information).
* Modular pipeline allows plugging in real telematics feeds.
* Extensions possible: gamification, weather/traffic APIs, real-time scoring.

---

📌 **Submission Archive:** `Sreekumar_Ajay_TelematicsUBI.zip` contains all code, models, docs, and this README.
