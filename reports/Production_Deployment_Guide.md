# Production Deployment Guide — AllLife Churn Model

**Scope**: End-to-end operationalisation of the XGBoost churn model on Databricks, including monitoring, alerting, retraining, and champion/challenger governance.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Drift Detection Concepts](#2-drift-detection-concepts)
   - 2.1 [Population Stability Index (PSI)](#21-population-stability-index-psi)
   - 2.2 [Kolmogorov-Smirnov Test (KS)](#22-kolmogorov-smirnov-test-ks)
   - 2.3 [When to Use PSI vs KS](#23-when-to-use-psi-vs-ks)
   - 2.4 [Data Drift vs Concept Drift vs Output Drift](#24-data-drift-vs-concept-drift-vs-output-drift)
3. [Performance Monitoring](#3-performance-monitoring)
4. [Retraining Pipeline](#4-retraining-pipeline)
5. [Validation Gate](#5-validation-gate)
6. [Champion/Challenger A/B Routing](#6-championchallenger-ab-routing)
7. [Feature Store](#7-feature-store)
8. [Databricks Lakehouse Monitoring](#8-databricks-lakehouse-monitoring)
9. [Databricks Workflow Orchestration](#9-databricks-workflow-orchestration)
10. [Implementation Priority](#10-implementation-priority)
11. [Glossary](#11-glossary)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  QUARTERLY BATCH PIPELINE                                       │
│                                                                 │
│  Databricks Delta Lake (raw)                                    │
│        │                                                        │
│        ▼                                                        │
│  Feature Store ──────────────────────────────────────────────► │
│        │                              Feature Serving           │
│        ▼                                                        │
│  Scoring Job (predict.py)  ◄── MLflow Model Registry (Champion)│
│        │                                                        │
│        ▼                                                        │
│  Predictions Delta Table ──► Campaign Engine (CRM)             │
│                                                                 │
│  Monitoring Layer                                               │
│        ├── PSI / KS on input features   (weekly)               │
│        ├── Score distribution drift     (weekly)               │
│        └── AUC / Recall on actuals      (quarter-end)          │
│                │                                                │
│                ▼                                                │
│  Alerting → Trigger retraining workflow                         │
└─────────────────────────────────────────────────────────────────┘
```

**Key principle**: the model is retrained at most once per quarter (aligned with the billing snapshot window). Monitoring runs more frequently to catch distribution shifts early.

---

## 2. Drift Detection Concepts

### 2.1 Population Stability Index (PSI)

**What it measures**: how much the distribution of a variable has shifted between a reference period (training data) and a current period (new scoring data).

**Formula**:

```
PSI = Σ [ (Actual_i − Expected_i) × ln(Actual_i / Expected_i) ]
```

Where:
- The variable is binned into *k* buckets (typically 10 deciles)
- `Expected_i` = proportion of training observations in bucket *i*
- `Actual_i` = proportion of current observations in bucket *i*

**Interpretation thresholds**:

| PSI Value | Interpretation | Action |
|---|---|---|
| < 0.10 | No significant shift | Monitor normally |
| 0.10 – 0.20 | Moderate shift | Investigate, increase monitoring frequency |
| > 0.20 | Major shift | Retrain or recalibrate |

**Why PSI?**
- Symmetric: catches both left-shifts and right-shifts equally
- Aggregates the entire distribution into one number — easy to threshold
- Well-established in financial services (Basel, IFRS 9)

**Python implementation**:

```python
import numpy as np

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between reference and current distributions."""
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-9  # include minimum
    bins[-1] += 1e-9

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    ref_pct = (ref_counts / ref_counts.sum()).clip(1e-9)
    cur_pct = (cur_counts / cur_counts.sum()).clip(1e-9)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)
```

**Apply to all features and to the model score**:

```python
psi_results = {
    col: compute_psi(train_df[col].values, scoring_df[col].values)
    for col in feature_columns + ["churn_probability"]
}
flagged = {k: v for k, v in psi_results.items() if v > 0.10}
```

---

### 2.2 Kolmogorov-Smirnov Test (KS)

**What it measures**: the maximum absolute difference between the empirical Cumulative Distribution Functions (CDFs) of two samples.

**Formula**:

```
KS = max | F_reference(x) − F_current(x) |
```

Where:
- `F_reference(x)` = proportion of reference observations ≤ x
- `F_current(x)` = proportion of current observations ≤ x

**Interpretation**:
- KS ranges from 0 (identical distributions) to 1 (completely separated)
- For large samples (> 10,000), even small KS values can be statistically significant — the *p-value* is more useful than the statistic alone
- Rule of thumb: KS > 0.10 combined with p < 0.05 warrants investigation

**Python implementation**:

```python
from scipy import stats

def compute_ks(reference: np.ndarray, current: np.ndarray) -> dict:
    """Two-sample KS test between reference and current."""
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return {"ks_statistic": ks_stat, "p_value": p_value}
```

---

### 2.3 When to Use PSI vs KS

| Criterion | PSI | KS |
|---|---|---|
| **Purpose** | Operational monitoring, single summary number | Statistical hypothesis test |
| **Output** | Magnitude (how much drift) | Test statistic + p-value (is drift significant?) |
| **Binning** | Required — sensitive to bin choice | Not required — exact |
| **Small samples** | Unreliable below ~1,000 obs | Valid for small samples |
| **Industry standard** | Credit risk, insurance, banking | Academic, A/B testing, model validation |
| **Use in production** | Primary drift KPI in dashboards | Secondary check, especially for smaller feature populations |

**Recommendation for AllLife**: use PSI as the primary weekly monitoring KPI (matches industry practice); use KS as a secondary validation check when PSI is in the 0.10–0.20 amber zone to determine whether the shift is statistically significant.

---

### 2.4 Data Drift vs Concept Drift vs Output Drift

Understanding the *type* of drift determines the right remediation.

```
┌────────────────────────────────────────────────────────────────┐
│  TYPE             │  WHAT CHANGES        │  SYMPTOM            │
├────────────────────────────────────────────────────────────────┤
│  Data drift       │  P(X) changes        │  PSI > 0.10 on      │
│  (covariate)      │  Feature distributions│  input features     │
│                   │  shift (e.g., new     │                     │
│                   │  tariff plan floods   │                     │
│                   │  the base)            │                     │
├────────────────────────────────────────────────────────────────┤
│  Concept drift    │  P(Y|X) changes      │  AUC / Recall drop  │
│                   │  Same features, new  │  with stable PSI    │
│                   │  churn drivers (e.g., │                     │
│                   │  new competitor enters│                     │
│                   │  market)              │                     │
├────────────────────────────────────────────────────────────────┤
│  Output drift     │  P(Ŷ) changes        │  Score distribution │
│  (label shift)    │  Model score dist.   │  PSI > 0.10;        │
│                   │  shifts — may be a   │  predicted churn    │
│                   │  downstream symptom  │  rate doubles       │
│                   │  of concept drift    │                     │
└────────────────────────────────────────────────────────────────┘
```

**How to diagnose**:
1. Check PSI on raw input features → if high: **data drift**
2. Check PSI on `churn_probability` output → if high with low feature PSI: **output drift / label shift**
3. Check AUC/Recall on actuals (available only at quarter-end after churn labels resolve) → if low with stable PSI: **concept drift**

---

## 3. Performance Monitoring

### Metrics to track

| Metric | Frequency | Trigger threshold | Notes |
|---|---|---|---|
| PSI (all features) | Weekly | > 0.10 amber, > 0.20 red | 10 decile bins on training distribution |
| PSI (score) | Weekly | > 0.10 amber, > 0.20 red | Reference = training score distribution |
| KS (top 5 features) | Weekly | > 0.10 + p < 0.05 | Secondary check on highest-importance features |
| ROC-AUC | Quarterly (after labels) | < 0.88 | Requires waiting for churn labels to resolve |
| Recall @ optimal threshold | Quarterly | < 0.80 | Recall matters most given asymmetric costs |
| Predicted churn rate | Weekly | ± 5pp from baseline | Quick proxy before labels are available |
| Campaign contact volume | Weekly | ± 20% from baseline | Downstream sanity check |

### Monitoring lag

Performance metrics (AUC, Recall) can only be computed **after churn labels resolve** — typically at end of quarter. This creates an inherent 90-day lag. Use proxy signals (PSI on scores, predicted churn rate) for earlier warning.

```
Quarter Q:  [Scoring] ──────────────────────────► [Labels available]
                │                                           │
                ▼                                           ▼
            PSI/KS (immediate)                     AUC/Recall (Q+1)
            proxy alerts                           true performance
```

---

## 4. Retraining Pipeline

### Trigger conditions

Retrain when **any** of the following:

1. PSI > 0.20 on any top-5 feature (by SHAP importance)
2. PSI > 0.20 on the model score distribution
3. ROC-AUC < 0.88 on a held-out validation set (lagged)
4. Recall < 0.80 at the operational threshold
5. Calendar-based: once per quarter regardless (scheduled refresh)

### Retraining steps

```python
# Pseudo-code for Databricks retraining job
def retrain_pipeline(new_data_path: str, mlflow_experiment: str):
    # 1. Load new data (extend training window)
    df = load_and_filter_population(new_data_path)

    # 2. Re-run feature engineering (same transformations)
    X, y = prepare_features(df)

    # 3. Optuna hyperparameter search (50 trials, 5-fold CV)
    best_params = run_optuna_study(X, y, n_trials=50)

    # 4. Train final model on full training set
    model = train_xgboost(X, y, best_params)

    # 5. Evaluate on holdout
    metrics = evaluate_on_holdout(model, X_test, y_test)

    # 6. Log to MLflow
    with mlflow.start_run(experiment_id=mlflow_experiment):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    # 7. Validation gate (see section 5)
    if passes_validation_gate(metrics):
        promote_to_staging(model)
```

---

## 5. Validation Gate

Before any retrained model is promoted to production (Champion), it must pass:

| Check | Threshold | Rationale |
|---|---|---|
| Test ROC-AUC | ≥ 0.88 | Minimum discriminatory power |
| Test Recall @ F2 threshold | ≥ 0.80 | Business requires catching most churners |
| AUC gap (CV vs Test) | < 0.03 | Overfitting guard |
| PSI vs previous champion | < 0.20 | Model scores should not shift dramatically |
| Net campaign value | ≥ $3,000 | Business value floor |

```python
def passes_validation_gate(metrics: dict) -> bool:
    checks = [
        metrics["roc_auc"] >= 0.88,
        metrics["recall"] >= 0.80,
        abs(metrics["cv_auc"] - metrics["test_auc"]) < 0.03,
        metrics["net_value"] >= 3000,
    ]
    if not all(checks):
        failed = [i for i, c in enumerate(checks) if not c]
        raise ValueError(f"Validation gate failed on checks: {failed}")
    return True
```

---

## 6. Champion/Challenger A/B Routing

### Why Champion/Challenger?

Retraining a model and immediately routing 100% of traffic carries risk. Champion/Challenger routes a small fraction to the new (Challenger) model, measures real-world performance, and promotes only on evidence.

### Deterministic routing by customer_id

```python
import hashlib

def get_model_assignment(customer_id: str, challenger_pct: float = 0.10) -> str:
    """Deterministic assignment: same customer always gets same model."""
    hash_val = int(hashlib.md5(customer_id.encode()).hexdigest(), 16)
    bucket = (hash_val % 100) / 100.0
    return "challenger" if bucket < challenger_pct else "champion"
```

**Why deterministic?** A customer who was scored by the Challenger last quarter should be scored by the same model this quarter to avoid attribution noise in campaign analysis.

### Promotion criteria

After one full quarter with the Challenger:

1. Challenger AUC ≥ Champion AUC − 0.005 (not significantly worse)
2. Challenger Recall ≥ Champion Recall − 0.02
3. Challenger Net Value ≥ Champion Net Value

If all three pass → promote Challenger to Champion, retire previous Champion.

```python
def evaluate_champion_challenger(champ_metrics: dict, chal_metrics: dict) -> str:
    if (
        chal_metrics["roc_auc"] >= champ_metrics["roc_auc"] - 0.005
        and chal_metrics["recall"] >= champ_metrics["recall"] - 0.02
        and chal_metrics["net_value"] >= champ_metrics["net_value"]
    ):
        return "promote_challenger"
    return "retain_champion"
```

---

## 7. Feature Store

### Purpose

Prevents **train/serve skew**: the most common source of silent model degradation in production. Train/serve skew occurs when the features computed at training time differ (even subtly) from those computed at scoring time — different SQL logic, different aggregation windows, different NULL handling.

### How it works in Databricks

```python
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# At training time: write features to store
fs.create_table(
    name="alllife.churn_features",
    primary_keys=["customer_id", "snapshot_date"],
    schema=features_df.schema,
    description="Churn model input features — quarterly snapshot"
)
fs.write_table(name="alllife.churn_features", df=features_df, mode="merge")

# At scoring time: read features from store (identical computation)
feature_lookups = [
    FeatureLookup(
        table_name="alllife.churn_features",
        lookup_key=["customer_id", "snapshot_date"]
    )
]
scoring_set = fs.create_training_set(
    df=customers_df,
    feature_lookups=feature_lookups,
    label=None
)
```

**Benefits**:
- Same feature computation code runs at both train and score time
- Automatic lineage tracking (which model used which features)
- Enables point-in-time correct feature lookups (no future leakage)

---

## 8. Databricks Lakehouse Monitoring

Databricks provides a native monitoring API (`databricks.sdk.service.catalog`) that computes PSI, KS, and performance metrics automatically for Delta tables.

### Setup

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries

w = WorkspaceClient()

w.quality_monitors.create(
    table_name="alllife.churn_predictions",
    assets_dir="/Volumes/alllife/monitoring/churn",
    output_schema_name="alllife_monitoring",
    time_series=MonitorTimeSeries(
        timestamp_col="snapshot_date",
        granularities=["1 week"]
    ),
    baseline_table_name="alllife.churn_predictions_training_baseline",
    slicing_exprs=["internet_type", "contract"],  # segment-level drift
)
```

### What it generates automatically

- **Profile table**: summary statistics per column per time window
- **Drift table**: PSI and KS per column, compared to baseline
- **Dashboard**: auto-generated Databricks dashboard with time-series charts

### Alerting

```python
# Databricks SQL alert on drift table
# SQL: SELECT * FROM alllife_monitoring.churn_predictions_drift
#       WHERE psi > 0.10 AND window_end_time = current_date()

# Configure alert to send email / Slack notification via webhook
```

---

## 9. Databricks Workflow Orchestration

Full quarterly pipeline as a Databricks Workflow (YAML definition):

```yaml
# databricks_workflow.yml
name: alllife_churn_quarterly

schedule:
  quartz_cron_expression: "0 0 6 1 1,4,7,10 ?"  # 6am on 1st of each quarter
  timezone_id: "Australia/Sydney"

tasks:
  - task_key: feature_engineering
    notebook_task:
      notebook_path: /Repos/alllife/churn/scripts/feature_engineering
    new_cluster:
      spark_version: "14.3.x-scala2.12"
      node_type_id: "i3.xlarge"
      num_workers: 4

  - task_key: drift_monitoring
    depends_on:
      - task_key: feature_engineering
    notebook_task:
      notebook_path: /Repos/alllife/churn/scripts/drift_monitoring
    new_cluster:
      spark_version: "14.3.x-scala2.12"
      node_type_id: "i3.xlarge"
      num_workers: 2

  - task_key: conditional_retrain
    depends_on:
      - task_key: drift_monitoring
    condition_task:
      op: "GREATER_THAN"
      left: "{{tasks.drift_monitoring.values.max_psi}}"
      right: "0.10"

  - task_key: retrain
    depends_on:
      - task_key: conditional_retrain
        outcome: "true"
    python_wheel_task:
      package_name: "telecom_churn"
      entry_point: "train"
    new_cluster:
      spark_version: "14.3.x-scala2.12"
      node_type_id: "i3.2xlarge"
      num_workers: 8

  - task_key: batch_scoring
    depends_on:
      - task_key: retrain
        outcome: "true"
      - task_key: drift_monitoring
        outcome: "false"  # also score even without retrain
    python_wheel_task:
      package_name: "telecom_churn"
      entry_point: "predict"
    new_cluster:
      spark_version: "14.3.x-scala2.12"
      node_type_id: "i3.xlarge"
      num_workers: 4

  - task_key: notify_campaign_team
    depends_on:
      - task_key: batch_scoring
    notebook_task:
      notebook_path: /Repos/alllife/churn/scripts/notify
```

---

## 10. Implementation Priority

| Priority | Component | Effort | Impact |
|---|---|---|---|
| **1** | PSI monitoring on score distribution | Low | High — earliest drift signal |
| **2** | Validation gate before promotion | Low | High — prevents silent degradation |
| **3** | MLflow Model Registry (Staging → Production) | Low | High — governance + rollback |
| **4** | Databricks Workflow (quarterly scheduler) | Medium | High — eliminates manual runs |
| **5** | Feature Store | Medium | Medium — prevents train/serve skew |
| **6** | Databricks Lakehouse Monitoring | Medium | Medium — native PSI/KS dashboard |
| **7** | Champion/Challenger routing | High | Medium — requires campaign attribution infra |
| **8** | Real-time scoring endpoint | High | Low for now (batch is sufficient) |

**Recommended first sprint** (2 weeks): items 1–4. These require no new Databricks infrastructure beyond what is already used for MLflow tracking.

---

## 11. Glossary

| Term | Definition |
|---|---|
| **PSI** | Population Stability Index. Quantifies how much a feature or score distribution has shifted between training and production. Single scalar, higher = more drift. |
| **KS (Kolmogorov-Smirnov)** | Non-parametric two-sample test measuring the maximum gap between two empirical CDFs. Provides a p-value unlike PSI. |
| **Data drift** | The distribution of input features P(X) changes over time, while the relationship P(Y\|X) stays the same. |
| **Concept drift** | The underlying relationship P(Y\|X) changes — same features now predict churn differently. Requires new labels to detect. |
| **Output drift** | The model's predicted score distribution P(Ŷ) shifts. Can be a symptom of either data or concept drift. |
| **Train/serve skew** | Features computed differently at training time vs scoring time — same column name, different logic. The Feature Store eliminates this. |
| **Champion/Challenger** | A/B testing framework where the current production model (Champion) and a candidate model (Challenger) each score a subset of customers. The Challenger is promoted only if it outperforms. |
| **Validation gate** | A set of minimum metric thresholds that a retrained model must pass before it can be promoted from Staging to Production in the Model Registry. |
| **F2 score** | F-beta score with β=2, weighting recall twice as heavily as precision. Used here because missing a churner ($180 lost margin) costs more than a wasted contact ($4). |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve. Threshold-invariant measure of a model's ability to rank churners above non-churners. |
| **Optuna TPE** | Tree-structured Parzen Estimator — Bayesian hyperparameter optimisation that builds a probabilistic model of which hyperparameters are likely to perform well. |
| **SHAP** | SHapley Additive exPlanations. Game-theory-based method assigning each feature a contribution to a specific prediction. Used for both global and local interpretability. |
| **MLflow Model Registry** | MLflow component for versioning models and managing their lifecycle: None → Staging → Production → Archived. |
| **Feature Store** | Databricks (or Feast) component that centralises feature computation and serves features consistently to both training and inference pipelines. |
| **Databricks Lakehouse Monitoring** | Native Databricks service that automatically computes statistical profiles, drift metrics (PSI/KS), and performance metrics for Delta tables on a schedule. |

---

*Generated as part of the AllLife Communications Senior Data Science Technical Assessment.*  
*Business parameters ($4 contact cost, $180 retention margin, 500 contact cap) are hypothetical values from the assessment brief and must be replaced with empirically validated figures before production deployment.*
