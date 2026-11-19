# AllLife Communications — Telecom Churn Prediction

**Senior Data Science Technical Assessment — Part A**

---

## Project Structure

```
telecom_churn/
├── config/
│   └── config.yaml              # Centralised config (paths, hyperparams, MLflow)
├── src/
│   ├── data/
│   │   └── loader.py            # SQLite reader, population filter, quality checks
│   ├── features/
│   │   └── engineering.py       # Feature engineering + sklearn ColumnTransformer
│   ├── eda/
│   │   └── analysis.py          # EDA functions and figure generation
│   ├── models/
│   │   └── trainer.py           # Optuna studies + MLflow tracking + model selection
│   └── evaluation/
│       ├── metrics.py           # ROC-AUC, F2, threshold optimisation, ROI calc
│       └── shap_analysis.py     # SHAP global + local explainability (TreeExplainer)
├── scripts/
│   ├── run_eda.py               # Standalone EDA runner
│   ├── train.py                 # Main training entry point
│   ├── evaluate.py              # Post-training comparison plots + SHAP analysis
│   └── predict.py               # Batch scoring: load champion, score new data
├── notebooks/
│   └── 01_EDA.py                # EDA notebook (Databricks-compatible)
├── reports/
│   ├── figures/                 # All generated plots (00–15)
│   ├── predictions.csv          # Batch scores from predict.py
│   ├── logistic_regression_odds_ratios.csv  # LR coefficients as odds ratios
│   └── Part_B_Answers.md        # Written reasoning exercise
├── tests/
│   ├── __init__.py
│   ├── test_features.py         # Unit tests for feature engineering
│   └── test_metrics.py          # Unit tests for evaluation metrics
├── requirements.txt
└── README.md
```

---

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) (`pip install uv` or `brew install uv`).

```bash
# 1. Install dependencies and create virtual environment
cd telecom_churn
uv sync --dev

# 2. Run EDA
uv run python scripts/run_eda.py

# 3. Train all models (30 Optuna trials each; ~8 min on MacBook M1)
uv run python scripts/train.py --trials 30

# 4. Full Optuna run (config default: 50 trials)
uv run python scripts/train.py

# 5. Generate evaluation plots + SHAP analysis
uv run python scripts/evaluate.py

# 6. Score customers (batch inference)
uv run python scripts/predict.py                          # scores SQLite population
uv run python scripts/predict.py --input customers.csv    # scores a custom CSV
uv run python scripts/predict.py --run-id <mlflow_run_id> # use a specific run

# 7. Run tests
uv run pytest tests/ -v

# 8. Launch MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Databricks Deployment

```python
# In a Databricks notebook or job:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
# (uv is not available in Databricks clusters; pip install from requirements.txt directly)

import os
os.environ["MLFLOW_TRACKING_URI"] = "databricks"  # Uses Databricks Managed MLflow

# Run training
%run ./scripts/train
```

Alternatively, use a Databricks Workflow with the `scripts/train.py` as a Python task.

---

## Key Design Decisions

### 1. Modelling Population: Exclude "Joined" Customers
The dataset contains 413 `Joined` customers (new this quarter). They are **excluded** from training because:
- They have not had sufficient tenure to exhibit churn behaviour
- Including them would create a systematic bias toward predicting `Stayed` for new customers — trivially true but unactionable for retention campaigns
- **Final population**: 5,986 customers (1,712 Churned / 4,274 Stayed)

### 2. Target Variable
Binary: `churn_label = 1` if `customer_status == 'Churned'`, else `0`.
**Churn rate: 28.6%** — moderate class imbalance, handled with `class_weight='balanced'` and `scale_pos_weight`.

### 3. Leakage Prevention
| Column | Disposition | Reason |
|---|---|---|
| `churn_category` | **EXCLUDED** | Post-churn outcome — hard leakage |
| `churn_reason` | **EXCLUDED** | Post-churn outcome — hard leakage |
| `customer_id`, `zip_code` | **EXCLUDED** | ID/location, no predictive signal as-is |
| `customer_status` | **EXCLUDED** | Label itself |
| `total_charges`, `total_revenue` | **RETAINED** | End-of-quarter snapshots — valid for batch scoring at Q-end; noted as `availability_window: quarter_end` in data contract |

### 4. Feature Engineering
Seven domain-driven derived features added before the sklearn pipeline:
- `is_short_tenure`: tenure < 12 months — most important predictor
- `is_fiber`: Fiber Optic internet — 42% churn rate vs 20% average
- `monthly_charges_ratio`: total_charges / tenure — spend velocity
- `net_revenue_per_month`: revenue per month of relationship
- `has_extra_data_charges`: exceeded plan at least once
- `streaming_count`: number of streaming services (bundle stickiness)
- `referrals_zero`: no referrals made — loyalty signal

### 5. Model Selection & Hyperparameter Optimisation
Optuna TPE sampler with 30–50 trials per model, 5-fold stratified cross-validation, maximising ROC-AUC:

| Model | CV ROC-AUC | Test ROC-AUC | Gap | Test F2 | Test Recall | Test Precision | Threshold | Net Value |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** ⭐ | 0.9040 | **0.9087** | +0.005 | 0.8111 | 0.9213 | 0.5486 | 0.290 | $3,384 |
| LightGBM | 0.9032 | 0.9064 | +0.003 | 0.8086 | 0.9359 | 0.5237 | 0.285 | $3,326 |
| Random Forest | 0.8969 | 0.9012 | +0.004 | 0.8006 | 0.8921 | 0.5677 | 0.375 | $3,352 |
| Logistic Regression | 0.8825 | 0.8881 | +0.006 | 0.8051 | 0.9155 | 0.5433 | 0.360 | $3,340 |

*Gap = Test AUC − CV AUC. All gaps are positive and < 0.006 — no evidence of overfitting in any model.*  
*Metrics at optimised F2 threshold (not default 0.5).*

**Why XGBoost?**
- Best ROC-AUC on held-out test set
- Gradient boosting captures non-linear interactions (tenure × internet_type) that LR misses
- Native handling of missing values (customers without internet service)
- Interpretable via feature importances; supports SHAP values

**Why not LightGBM?** Only 0.0023 ROC-AUC gap at equivalent regularization. LightGBM has higher recall (0.9359 vs 0.9213) but lower precision, resulting in $58 less net campaign value and 11pp lower ROI. At Databricks scale (millions of rows) LightGBM would be preferred for training speed.

### 6. Evaluation Metric: Primary ROC-AUC, Secondary F2 (β=2)
- **ROC-AUC**: threshold-invariant, handles imbalance, enables ranking customers by risk
- **F2 score**: weights recall 2× over precision, reflecting the business asymmetry: missing a churner ($180 lost margin) is more costly than a wasted contact ($4 cost)
- Break-even at 10% response rate: P(churn) × 10% × $180 > $4 → P(churn) > 22%

### 7. Threshold Optimisation
Default threshold (0.5) gives balanced trade-off. Optimised F2 threshold (~0.285 for XGBoost) increases recall to ~93.9% at the cost of lower precision (~52.4%). Business should choose based on campaign budget and response rate estimates.

### 8. Class Imbalance Strategy
- `class_weight='balanced'` for RF and LR
- `scale_pos_weight = 2.5` (ratio of negatives to positives) for XGBoost and LightGBM
- **Rejected SMOTE**: generates synthetic samples by interpolation, which can produce unrealistic customers when categorical features are mixed with continuous ones. The class_weight approach is simpler, more transparent, and competitive in performance.

---

## Assumptions

### Business parameters (provided in Part B problem statement)
These values are **explicitly hypothetical** — stated as such in the assessment brief. They should be replaced with empirically validated figures before any production deployment.

| Parameter | Value | Source |
|---|---|---|
| Cost per contact | $4 | Part B problem statement |
| Contribution margin per incremental retention | $180 | Part B problem statement |
| Maximum campaign contacts | 500 | Part B problem statement |

### Modelling assumptions (not in the problem statement)
These were added during modelling and require validation before production use.

| Assumption | Value | Rationale | How to validate |
|---|---|---|---|
| Campaign response rate | 10% | Conservative industry benchmark for telecom retention campaigns | Run the RCT described in Part B Q1; measure observed retention rate per decile |
| Scoring window | Quarter-end batch | Cumulative billing fields (`total_charges`, `total_revenue`) reflect end-of-quarter totals — only valid at Q-end | For intra-quarter scoring, replace with rolling partial-period aggregates |
| Modelling population | Churned + Stayed only | `Joined` customers excluded — insufficient tenure to exhibit stable churn behaviour | Re-evaluate if the business needs to score new customers separately |
| Class imbalance handling | `class_weight='balanced'` + `scale_pos_weight` | SMOTE rejected: interpolation on mixed categorical/numeric features produces unrealistic synthetic customers | Benchmark SMOTE on a held-out validation set if imbalance worsens in future quarters |

---

## EDA Key Findings

1. **Tenure is the strongest predictor**: 0-6 month customers churn at 77.7%; 60+ month customers at 6.9%
2. **Fiber Optic drives double the churn rate** (42%) vs DSL (21%) — competitive pressure from ISPs
3. **Top churn reasons** (EDA only): "Competitor had better devices" (286), "Competitor made better offer" (276) — price/product competition dominates
4. **Number of referrals is strongly protective**: zero-referral customers churn at ~40%; 5+ referrals customers at <5%
5. **Payment method signal**: Bank withdrawal customers churn more than Credit Card customers — may indicate lower account integration
6. **No missing values in predictors** (after excluding internet-service conditional fields handled by imputation in pipeline)

---

## MLflow Tracking
Experiments logged to `sqlite:///mlflow.db` (local dev) or Databricks Managed MLflow (production).

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Navigate to http://localhost:5000
```

Each run logs:
- Hyperparameters (best Optuna params)
- Metrics: ROC-AUC, F2, recall, precision, net campaign value
- Feature importance CSV artifact
- Trained sklearn Pipeline (pickle format)

---

## SHAP Explainability

`evaluate.py` runs SHAP after model comparison and generates two figures for the champion:

| Figure | Description |
|---|---|
| `13_shap_importance_<model>.png` | Bar plot — mean \|SHAP\| per feature (global ranking) |
| `14_shap_beeswarm_<model>.png` | Beeswarm — feature value vs SHAP magnitude per customer |

`TreeExplainer` is used for XGBoost / LightGBM / RandomForest (exact, fast); `LinearExplainer` for Logistic Regression. SHAP complements the tree feature importances by showing both direction (positive/negative effect) and distribution across customers — answering *why* a specific customer is flagged, not just which features matter overall.

---

## Batch Scoring

`scripts/predict.py` loads the MLflow champion and scores a dataset:

```bash
python scripts/predict.py --input new_customers.csv --output scores.csv
```

Output columns:

| Column | Description |
|---|---|
| `customer_id` | Customer identifier |
| `churn_probability` | Model score in [0, 1] |
| `churn_prediction` | 1 if probability ≥ optimal threshold |
| `risk_tier` | High (≥0.6) / Medium (0.3–0.6) / Low (<0.3) |

---

## Testing

```bash
uv run pytest tests/ -v
```

29 unit tests across two modules:

| Module | Coverage |
|---|---|
| `tests/test_features.py` | `add_derived_features`, `encode_binary_columns`, `prepare_features` — leakage prevention, derived columns, binary encoding, train/test partition |
| `tests/test_metrics.py` | `compute_all_metrics`, `find_optimal_threshold`, `compute_net_value` — perfect model AUC=1, confusion matrix integrity, threshold strategies, ROI arithmetic |

---

## Limitations and Future Work

| Limitation | Mitigation |
|---|---|
| No causal estimate of campaign lift | Pilot RCT (see Part B Q1) |
| Quarter-end snapshot — no event history | Add rolling event features when available |
| No customer lifetime value weighting | Weight loss by predicted CLV in objective |
| No real-time scoring | Add Databricks Model Serving endpoint |
| Fairness not formally tested | Add demographic parity checks (Part B Q5) |
