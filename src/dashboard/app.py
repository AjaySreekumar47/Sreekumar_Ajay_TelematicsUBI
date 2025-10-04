
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, roc_curve, precision_recall_curve
)

# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="Telematics UBI Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
BRONZE_DIR = f"/content/Sreekumar_Ajay_TelematicsUBI/data/bronze"
SILVER_FILE = f"/content/Sreekumar_Ajay_TelematicsUBI/data/silver/silver_events.parquet"
GOLD_FILE = f"/content/Sreekumar_Ajay_TelematicsUBI/data/gold/driver_features.parquet"
TRAIN_FILE = f"/content/Sreekumar_Ajay_TelematicsUBI/data/training/training_set.parquet"
DLQ_FILE = f"/content/Sreekumar_Ajay_TelematicsUBI/data/dlq/events_dlq.jsonl"
MODELS_DIR = f"/content/Sreekumar_Ajay_TelematicsUBI/models"
OUT_DIR = f"/content/Sreekumar_Ajay_TelematicsUBI/data/training"

# Prediction files from Phase 6
PRED_FILES = {
    "Logistic Regression": "predictions_logit.parquet",
    "Random Forest": "predictions_rf.parquet",
    "XGBoost": "predictions_xgb.parquet",
    "Linear Regression (Severity)": "predictions_linreg.parquet",
    "XGBoost (Severity)": "predictions_xgb_severity.parquet",
    "Gamma GLM (Severity)": "predictions_gamma_glm.parquet"
}

# Metrics files
METRICS_FILES = {
    "Logistic Regression": "logit_metrics.json",
    "Random Forest": "rf_metrics.json",
    "XGBoost": "xgb_metrics.json",
    "Linear Regression (Severity)": "linreg_severity_metrics.json",
    "XGBoost (Severity)": "xgb_severity_metrics.json",
    "Gamma GLM (Severity)": "gamma_glm_metrics.json"
}

# ============================================================================
# Utility Functions
# ============================================================================

@st.cache_data
def safe_read_parquet(path):
    """Safely read parquet file with error handling."""
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Error reading {path}: {e}")
    return None

@st.cache_data
def safe_read_json(path):
    """Safely read JSON file with error handling."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Error reading {path}: {e}")
    return None

def get_file_info(directory):
    """Get metadata about files in a directory."""
    if not os.path.exists(directory):
        return pd.DataFrame()

    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            files.append({
                "filename": filename,
                "size_kb": stat.st_size / 1024,
                "modified": pd.to_datetime(stat.st_mtime, unit="s")
            })

    return pd.DataFrame(files).sort_values("modified", ascending=False) if files else pd.DataFrame()

def count_jsonl_lines(filepath):
    """Count lines in a JSONL file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return sum(1 for _ in f)
    except:
        pass
    return 0

# ============================================================================
# Header
# ============================================================================

st.title("🚗 Telematics Usage-Based Insurance Dashboard")
st.markdown("### Production POC Monitoring & Analytics")
st.markdown("---")

# ============================================================================
# Sidebar Navigation
# ============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Overview", "📥 Data Ingestion", "🔧 Feature Engineering",
     "🤖 Model Performance", "📈 Model Comparison", "🔍 Model Registry"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

# Quick status checks
training_exists = os.path.exists(TRAIN_FILE)
models_exist = os.path.exists(os.path.join(MODELS_DIR, "registry.json"))

st.sidebar.success("✓ Training Data" if training_exists else "✗ Training Data")
st.sidebar.success("✓ Models Trained" if models_exist else "✗ Models Trained")

# ============================================================================
# PAGE: Overview
# ============================================================================

if page == "📊 Overview":
    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Data pipeline metrics
    bronze_files = get_file_info(BRONZE_DIR)
    with col1:
        st.metric("Bronze Files", len(bronze_files))

    with col2:
        silver_df = safe_read_parquet(SILVER_FILE)
        st.metric("Silver Records", f"{len(silver_df):,}" if silver_df is not None else "N/A")

    with col3:
        gold_df = safe_read_parquet(GOLD_FILE)
        st.metric("Drivers (Gold)", f"{len(gold_df):,}" if gold_df is not None else "N/A")

    with col4:
        dlq_count = count_jsonl_lines(DLQ_FILE)
        st.metric("DLQ Records", dlq_count, delta="Issues" if dlq_count > 0 else None)

    st.markdown("---")

    # Training set summary
    st.subheader("Training Set Summary")
    train_df = safe_read_parquet(TRAIN_FILE)

    if train_df is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Samples", f"{len(train_df):,}")

        # --- Flexible column mapping ---
        claim_col = None
        severity_col = None

        if "had_claim_within_30d" in train_df.columns:
            claim_col = "had_claim_within_30d"
        elif "claim_occurred" in train_df.columns:
            claim_col = "claim_occurred"

        if "claim_severity" in train_df.columns:
            severity_col = "claim_severity"
        elif "claim_amount" in train_df.columns:
            severity_col = "claim_amount"

        if claim_col:
            with col2:
                claim_rate = train_df[claim_col].mean()
                st.metric("Claim Rate", f"{claim_rate:.2%}")

            claimants = train_df[train_df[claim_col] == 1]

            with col3:
                st.metric("Total Claims", f"{len(claimants):,}")

            with col4:
                if severity_col and len(claimants) > 0:
                    avg_severity = claimants[severity_col].mean()
                    st.metric("Avg Severity", f"${avg_severity:,.0f}")
                else:
                    st.metric("Avg Severity", "N/A")
        else:
            st.warning("⚠️ No claim indicator column found in training set")

    else:
        st.warning("Training data not found. Run Phase 5 to generate training set.")

    st.markdown("---")

    # Model registry summary
    st.subheader("Trained Models")
    registry = safe_read_json(os.path.join(MODELS_DIR, "registry.json"))

    if registry:
        registry_df = pd.DataFrame(registry)
        st.dataframe(
            registry_df[["name", "version", "created_at"]].tail(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No models registered yet. Run Phase 6 to train models.")
    
        st.markdown("---")

    # Premium Pricing Summary
    st.subheader("Premium Pricing (Pricing Engine)")

    premiums_file = os.path.join(OUT_DIR, "driver_premiums.parquet")
    premiums_df = safe_read_parquet(premiums_file)

    if premiums_df is not None and len(premiums_df) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Drivers Priced", f"{premiums_df['driver_id'].nunique():,}")

        with col2:
            avg_premium = premiums_df["premium"].mean()
            st.metric("Average Premium", f"${avg_premium:,.2f}")

        with col3:
            high_risk = (premiums_df["premium"] > premiums_df["premium"].median() * 1.5).sum()
            st.metric("High-Risk Drivers", high_risk)

        if st.checkbox("Show sample premiums"):
            st.dataframe(premiums_df.head(20), use_container_width=True)
    else:
        st.info("No premiums file found. Run the Pricing Engine (Phase 7) to generate premiums.")

# ============================================================================
# PAGE: Data Ingestion
# ============================================================================

elif page == "📥 Data Ingestion":
    st.header("Data Ingestion Health")

    # Bronze layer
    st.subheader("Bronze Layer (Raw Events)")
    bronze_files = get_file_info(BRONZE_DIR)

    if not bronze_files.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(bronze_files, use_container_width=True, hide_index=True)

        with col2:
            st.metric("Total Files", len(bronze_files))
            st.metric("Total Size", f"{bronze_files['size_kb'].sum():.1f} KB")
            st.metric("Latest File", bronze_files.iloc[0]["filename"])

        # Timeline visualization
        fig = px.line(
            bronze_files.sort_values("modified"),
            x="modified",
            y="size_kb",
            markers=True,
            title="File Ingestion Timeline"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bronze files found. Run ingestion to populate data.")

    st.markdown("---")

    # Silver layer
    st.subheader("Silver Layer (Validated Events)")
    silver_df = safe_read_parquet(SILVER_FILE)

    if silver_df is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Events", f"{len(silver_df):,}")

        with col2:
            unique_drivers = silver_df["driver_id"].nunique()
            st.metric("Unique Drivers", f"{unique_drivers:,}")

        with col3:
            if "event_timestamp" in silver_df.columns:
                date_range = (silver_df["event_timestamp"].max() - silver_df["event_timestamp"].min()).days
                st.metric("Date Range (days)", date_range)

        # Event type distribution
        if "event_type" in silver_df.columns:
            event_dist = silver_df["event_type"].value_counts()
            fig = px.bar(
                x=event_dist.index,
                y=event_dist.values,
                title="Event Type Distribution",
                labels={"x": "Event Type", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Silver data not found. Run Phase 3 to process events.")

    st.markdown("---")

    # Dead Letter Queue
    st.subheader("Dead Letter Queue (DLQ)")
    dlq_count = count_jsonl_lines(DLQ_FILE)

    if dlq_count > 0:
        st.warning(f"⚠️ {dlq_count} records in DLQ require attention")

        if st.button("Show DLQ Sample"):
            try:
                with open(DLQ_FILE, "r") as f:
                    lines = [json.loads(line) for line in list(f)[:10]]
                st.json(lines)
            except Exception as e:
                st.error(f"Error reading DLQ: {e}")
    else:
        st.success("✓ No records in DLQ")

# ============================================================================
# PAGE: Feature Engineering
# ============================================================================

elif page == "🔧 Feature Engineering":
    st.header("Feature Engineering (Gold Layer)")

    gold_df = safe_read_parquet(GOLD_FILE)

    if gold_df is None:
        st.warning("Gold features not found. Run Phase 4 to generate features.")
    else:
        st.success(f"✓ Loaded {len(gold_df):,} driver profiles")

        # Feature statistics
        st.subheader("Feature Statistics")

        numeric_cols = gold_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            feature_stats = gold_df[numeric_cols].describe().T
            feature_stats["missing"] = gold_df[numeric_cols].isna().sum()
            st.dataframe(feature_stats, use_container_width=True)

        st.markdown("---")

        # Feature distributions
        st.subheader("Feature Distributions")

        feature_to_plot = st.selectbox(
            "Select feature to visualize",
            options=numeric_cols.tolist()
        )

        if feature_to_plot:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    gold_df,
                    x=feature_to_plot,
                    nbins=30,
                    title=f"Distribution: {feature_to_plot}"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(
                    gold_df,
                    y=feature_to_plot,
                    title=f"Box Plot: {feature_to_plot}"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Feature correlations
        st.subheader("Feature Correlations")

        if st.checkbox("Show correlation heatmap"):
            corr_cols = st.multiselect(
                "Select features (max 10)",
                options=numeric_cols.tolist(),
                default=numeric_cols.tolist()[:5]
            )

            if len(corr_cols) > 1:
                corr_matrix = gold_df[corr_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Raw data preview
        if st.checkbox("Show raw feature data"):
            st.dataframe(gold_df.head(100), use_container_width=True)

# ============================================================================
# PAGE: Model Performance
# ============================================================================

elif page == "🤖 Model Performance":
    st.header("Model Performance Details")

    model_type = st.radio(
        "Select model type",
        ["Frequency (Classification)", "Severity (Regression)"]
    )

    if model_type == "Frequency (Classification)":
        st.subheader("Frequency Models: Claim Prediction")

        freq_models = ["Logistic Regression", "Random Forest", "XGBoost"]
        selected_model = st.selectbox("Select model", freq_models)

        # Load predictions and metrics
        pred_file = PRED_FILES.get(selected_model)
        metrics_file = METRICS_FILES.get(selected_model)

        if pred_file and metrics_file:
            pred_path = os.path.join(OUT_DIR, pred_file)
            metrics_path = os.path.join(OUT_DIR, metrics_file)

            preds_df = safe_read_parquet(pred_path)
            metrics = safe_read_json(metrics_path)

            if preds_df is not None and metrics is not None:
                # Display metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("AUC-ROC", f"{metrics.get('AUC', 0):.4f}")
                    if metrics.get('AUC_CI'):
                        ci = metrics['AUC_CI']
                        st.caption(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

                with col2:
                    st.metric("Average Precision", f"{metrics.get('PR', 0):.4f}")
                    if metrics.get('PR_CI'):
                        ci = metrics['PR_CI']
                        st.caption(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

                with col3:
                    st.metric("Brier Score", f"{metrics.get('Brier', 0):.4f}")

                st.markdown("---")

                # ROC Curve
                y_true = preds_df["y_true"].values
                y_pred = preds_df["y_pred"].values

                fpr, tpr, _ = roc_curve(y_true, y_pred)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC={metrics.get("AUC", 0):.3f})',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    width=600, height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Prediction distribution
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.histogram(
                        preds_df,
                        x="y_pred",
                        color="y_true",
                        nbins=30,
                        title="Prediction Distribution by True Label"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Calibration bins
                    preds_df['pred_bin'] = pd.cut(preds_df['y_pred'], bins=10)
                    calib_df = preds_df.groupby('pred_bin', observed=True).agg({
                        'y_true': 'mean',
                        'y_pred': 'mean'
                    }).reset_index()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=calib_df['y_pred'],
                        y=calib_df['y_true'],
                        mode='markers+lines',
                        name='Model'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Perfect Calibration',
                        line=dict(dash='dash')
                    ))
                    fig.update_layout(
                        title="Calibration Plot",
                        xaxis_title="Predicted Probability",
                        yaxis_title="Actual Frequency"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Sample predictions
                if st.checkbox("Show prediction samples"):
                    st.dataframe(preds_df.head(50), use_container_width=True)
            else:
                st.warning(f"Predictions or metrics not found for {selected_model}")

    else:  # Severity models
        st.subheader("Severity Models: Claim Amount Prediction")

        sev_models = ["Linear Regression (Severity)", "XGBoost (Severity)", "Gamma GLM (Severity)"]
        selected_model = st.selectbox("Select model", sev_models)

        pred_file = PRED_FILES.get(selected_model)
        metrics_file = METRICS_FILES.get(selected_model)

        if pred_file and metrics_file:
            pred_path = os.path.join(OUT_DIR, pred_file)
            metrics_path = os.path.join(OUT_DIR, metrics_file)

            preds_df = safe_read_parquet(pred_path)
            metrics = safe_read_json(metrics_path)

            if preds_df is not None and metrics is not None:
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("MAE", f"${metrics.get('MAE', 0):,.2f}")

                with col2:
                    st.metric("RMSE", f"${metrics.get('RMSE', 0):,.2f}")

                with col3:
                    st.metric("R²", f"{metrics.get('R2', 0):.4f}")

                with col4:
                    st.metric("Norm Gini", f"{metrics.get('NormGini', 0):.4f}")

                st.markdown("---")

                # Actual vs Predicted
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.scatter(
                        preds_df,
                        x="y_true",
                        y="y_pred",
                        title="Actual vs Predicted Severity",
                        labels={"y_true": "Actual", "y_pred": "Predicted"},
                        trendline="ols"
                    )
                    fig.add_trace(go.Scatter(
                        x=[preds_df['y_true'].min(), preds_df['y_true'].max()],
                        y=[preds_df['y_true'].min(), preds_df['y_true'].max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    residuals = preds_df['y_true'] - preds_df['y_pred']
                    fig = px.histogram(
                        residuals,
                        nbins=30,
                        title="Residuals Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Sample predictions
                if st.checkbox("Show prediction samples"):
                    st.dataframe(preds_df.head(50), use_container_width=True)
            else:
                st.warning(f"Predictions or metrics not found for {selected_model}")

# ============================================================================
# PAGE: Model Comparison
# ============================================================================

elif page == "📈 Model Comparison":
    st.header("Model Comparison")

    # Frequency models comparison
    st.subheader("Frequency Models")

    freq_models = ["Logistic Regression", "Random Forest", "XGBoost"]
    freq_metrics = []

    for model in freq_models:
        metrics_file = METRICS_FILES.get(model)
        if metrics_file:
            metrics_path = os.path.join(OUT_DIR, metrics_file)
            metrics = safe_read_json(metrics_path)
            if metrics:
                freq_metrics.append({
                    "Model": model,
                    "AUC": metrics.get("AUC", np.nan),
                    "Avg Precision": metrics.get("PR", np.nan),
                    "Brier": metrics.get("Brier", np.nan),
                    "Train Size": metrics.get("n_train", np.nan),
                    "Test Size": metrics.get("n_test", np.nan)
                })

    if freq_metrics:
        freq_df = pd.DataFrame(freq_metrics)
        st.dataframe(freq_df.round(4), use_container_width=True, hide_index=True)

        # Visual comparison
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("AUC-ROC", "Average Precision", "Brier Score")
        )

        fig.add_trace(
            go.Bar(x=freq_df["Model"], y=freq_df["AUC"], name="AUC"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=freq_df["Model"], y=freq_df["Avg Precision"], name="AP"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=freq_df["Model"], y=freq_df["Brier"], name="Brier"),
            row=1, col=3
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No frequency model metrics found. Run Phase 6 to train models.")

    st.markdown("---")

    # Severity models comparison
    st.subheader("Severity Models")

    sev_models = ["Linear Regression (Severity)", "XGBoost (Severity)", "Gamma GLM (Severity)"]
    sev_metrics = []

    for model in sev_models:
        metrics_file = METRICS_FILES.get(model)
        if metrics_file:
            metrics_path = os.path.join(OUT_DIR, metrics_file)
            metrics = safe_read_json(metrics_path)
            if metrics:
                sev_metrics.append({
                    "Model": model,
                    "MAE": metrics.get("MAE", np.nan),
                    "RMSE": metrics.get("RMSE", np.nan),
                    "R²": metrics.get("R2", np.nan),
                    "Norm Gini": metrics.get("NormGini", np.nan),
                    "Claimants": metrics.get("n_claimants", np.nan)
                })

    if sev_metrics:
        sev_df = pd.DataFrame(sev_metrics)
        st.dataframe(sev_df.round(4), use_container_width=True, hide_index=True)

        # Visual comparison
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("MAE", "RMSE", "Normalized Gini")
        )

        fig.add_trace(
            go.Bar(x=sev_df["Model"], y=sev_df["MAE"], name="MAE"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=sev_df["Model"], y=sev_df["RMSE"], name="RMSE"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=sev_df["Model"], y=sev_df["Norm Gini"], name="Gini"),
            row=1, col=3
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No severity model metrics found. Run Phase 6 to train models.")

# ============================================================================
# PAGE: Model Registry
# ============================================================================

elif page == "🔍 Model Registry":
    st.header("Model Registry")

    registry_path = os.path.join(MODELS_DIR, "registry.json")
    registry = safe_read_json(registry_path)

    if registry:
        st.success(f"✓ {len(registry)} models registered")

        registry_df = pd.DataFrame(registry)

        # Format the display
        display_cols = ["name", "version", "data_hash", "created_at", "phase"]
        if all(col in registry_df.columns for col in display_cols):
            st.dataframe(
                registry_df[display_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(registry_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Model details
        st.subheader("Model Details")
        model_names = registry_df["name"].unique().tolist()
        selected = st.selectbox("Select model to inspect", model_names)

        if selected:
            model_info = registry_df[registry_df["name"] == selected].iloc[-1].to_dict()

            col1, col2 = st.columns(2)

            with col1:
                st.json({k: v for k, v in model_info.items() if k != "artifact"})

            with col2:
                # Check if artifact exists
                artifact_path = model_info.get("artifact")
                if artifact_path and os.path.exists(artifact_path):
                    st.success(f"✓ Model artifact exists")
                    st.code(artifact_path)

                    file_size = os.path.getsize(artifact_path) / 1024
                    st.metric("Artifact Size", f"{file_size:.1f} KB")
                else:
                    st.warning("Model artifact not found")
    else:
        st.info("No models registered yet. Run Phase 6 to train and register models.")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("**Telematics UBI POC Dashboard** | Phase 7 | Production Ready")
