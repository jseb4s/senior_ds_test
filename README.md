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
│   ├── models/
│   │   └── trainer.py           # Optuna studies + MLflow tracking + model selection
│   └── evaluation/
│       ├── metrics.py           # ROC-AUC, F2, threshold optimisation, ROI calc
│       └── shap_analysis.py     # SHAP global + local explainability (TreeExplainer)
├── scripts/
│   ├── train.py                 # Main training entry point
│   ├── evaluate.py              # Post-training comparison plots + SHAP analysis
│   ├── predict.py               # Batch scoring: load champion, score new data
│   └── register_model.py        # Manual step: register champion in MLflow Model Registry
├── notebooks/
│   └── EDA.ipynb                # Exploratory data analysis notebook
├── reports/
│   ├── figures/                 # All generated plots (00–15)
│   ├── predictions.csv          # Batch scores from predict.py
│   └── logistic_regression_odds_ratios.csv  # LR coefficients as odds ratios
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

# 2. Run EDA (open and execute the notebook)
# notebooks/EDA.ipynb

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

# 8. Register champion model in MLflow Model Registry (manual step — run after reviewing metrics)
uv run python scripts/register_model.py                   # registers best run by ROC-AUC
uv run python scripts/register_model.py --run-id <id>     # registers a specific run

# 9. Launch MLflow UI
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

### 3. Variable Inventory

Complete disposition of all 27 columns from the source database.

#### Excluded variables

| Column | Reason |
|---|---|
| `churn_category` | **Hard leakage** — post-churn outcome, only exists for customers who already left. Including it would give AUC ≈ 1.0 in training but the column is unavailable at scoring time. |
| `churn_reason` | **Hard leakage** — same as above. |
| `customer_status` | **Label itself** — the binary target `churn_label` is derived from this column; it cannot also be a predictor. |
| `customer_id` | **Identifier** — no predictive signal; would cause the model to memorise individual customers rather than learn patterns. |
| `zip_code` | **High-cardinality location ID** — hundreds of unique values, most with very few customers. `zip_population` is retained as a continuous proxy for urbanisation. |

#### Variables used directly (numeric)

| Column | Why retained |
|---|---|
| `age` | Customer age — older customers tend to have longer tenure and lower churn. |
| `number_of_referrals` | Strong protective signal — zero-referral customers churn at ~40% vs <5% for 5+ referrals. Used as-is AND as the basis for `referrals_zero`. |
| `tenure_in_months` | Most predictive raw feature — 0–6 month customers churn at 77.7%. Used as-is AND as the basis for `is_short_tenure`. |
| `avg_monthly_long_distance_charges` | Proxy for phone plan depth — higher spend = more engaged customer. |
| `avg_monthly_gb_download` | Internet usage intensity — heavy users may be more sensitive to competitor offers. |
| `total_charges` | Cumulative spend — reflects relationship value. Quarter-end snapshot only (see leakage note in Assumptions). |
| `total_refunds` | Negative signal — refunds indicate billing friction or service issues. |
| `total_extra_data_charges` | Overage charges — used as-is AND as the basis for `has_extra_data_charges`. |
| `total_long_distance_charges` | Cumulative phone spend — contributes to total revenue picture. |
| `total_revenue` | Net revenue including all charges. Quarter-end snapshot only. |
| `zip_population` | Urbanisation proxy — competitive intensity may differ between dense urban and rural markets. |

#### Variables used directly (binary — encoded Yes/No → 1/0)

| Column | Why retained |
|---|---|
| `married` | Married customers may have more stable household finances — lower churn tendency. |
| `phone_service` | Service subscription indicator. |
| `multiple_lines` | Having multiple lines increases switching cost. |
| `internet_service` | Whether the customer has internet — gate for internet-related features. |
| `streaming_tv` | Streaming subscription — contributes to `streaming_count`. |
| `streaming_movies` | Streaming subscription — contributes to `streaming_count`. |
| `streaming_music` | Streaming subscription — contributes to `streaming_count`. |
| `unlimited_data` | Unlimited plan customers are less likely to have overage friction. |
| `gender` | Retained as a demographic variable (Male → 1, Female → 0). No strong prior that gender predicts churn but included for the model to evaluate. |

#### Variables used directly (categorical — OrdinalEncoder for trees, OHE for LR)

| Column | Why retained |
|---|---|
| `internet_type` | Fiber Optic customers churn at 42% vs 20% average — one of the strongest categorical signals. Also used as the basis for `is_fiber`. |
| `payment_method` | Bank withdrawal customers show higher churn than credit card — may indicate lower account integration or financial instability. |

#### Derived features (created from existing variables)

| Feature | Formula | Rationale |
|---|---|---|
| `is_short_tenure` | `tenure_in_months < 12` → 0/1 | Captures the non-linear churn spike in the first year. A continuous tenure variable treats the difference between month 1 and month 11 the same as between month 50 and month 60 — this binary flag gives the model an explicit handle on the high-risk early period. |
| `is_fiber` | `internet_type == "Fiber Optic"` → 0/1 | Fiber customers churn at 2× the average rate. Extracting this as a binary feature ensures tree models can easily split on it without navigating the full cardinality of `internet_type`. |
| `monthly_charges_ratio` | `total_charges / tenure_in_months` | Spend velocity — normalises total charges by tenure to get a monthly average. A customer with $500 in charges after 2 months is very different from one with $500 after 50 months. |
| `net_revenue_per_month` | `total_revenue / tenure_in_months` | Revenue per month net of refunds — a cleaner profitability signal than raw charges. |
| `has_extra_data_charges` | `total_extra_data_charges > 0` → 0/1 | Whether the customer ever exceeded their data plan. Overage charges create billing friction and dissatisfaction — a binary flag is more interpretable than the raw dollar amount. |
| `streaming_count` | Sum of streaming_tv + streaming_movies + streaming_music | Bundle stickiness proxy — the more services a customer uses, the higher the switching cost. Collapses three correlated binary columns into one ordinal feature (0–3). |
| `referrals_zero` | `number_of_referrals == 0` → 0/1 | Customers who have never referred anyone show ~40% churn rate. Complements the continuous `number_of_referrals` by giving the model an explicit flag for the zero-referral segment. |
| `revenue_to_charges_ratio` | `total_revenue / total_charges` | Ratio of revenue to charges — captures whether the customer generates additional revenue (e.g. long distance) beyond the base plan. Values > 1 indicate a more engaged, higher-value customer. |

### 4. Leakage Prevention — Summary
| Column | Disposition | Reason |
|---|---|---|
| `churn_category` | **EXCLUDED** | Post-churn outcome — hard leakage |
| `churn_reason` | **EXCLUDED** | Post-churn outcome — hard leakage |
| `customer_id`, `zip_code` | **EXCLUDED** | ID/location, no predictive signal as-is |
| `customer_status` | **EXCLUDED** | Label itself |
| `total_charges`, `total_revenue` | **RETAINED** | End-of-quarter snapshots — valid for batch scoring at Q-end; noted as `availability_window: quarter_end` in data contract |

### 5. Feature Engineering
Eight domain-driven derived features added before the sklearn pipeline:
- `is_short_tenure`: tenure < 12 months — most important predictor
- `is_fiber`: Fiber Optic internet — 42% churn rate vs 20% average
- `monthly_charges_ratio`: total_charges / tenure — spend velocity
- `net_revenue_per_month`: revenue per month of relationship
- `has_extra_data_charges`: exceeded plan at least once
- `streaming_count`: number of streaming services (bundle stickiness)
- `referrals_zero`: no referrals made — loyalty signal
- `revenue_to_charges_ratio`: revenue vs charges ratio — customer value signal

### 6. Model Selection & Hyperparameter Optimisation
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

### 7. Evaluation Metric: Primary ROC-AUC, Secondary F2 (β=2)
- **ROC-AUC**: threshold-invariant, handles imbalance, enables ranking customers by risk
- **F2 score**: weights recall 2× over precision, reflecting the business asymmetry: missing a churner ($180 lost margin) is more costly than a wasted contact ($4 cost)
- Break-even at 10% response rate: P(churn) × 10% × $180 > $4 → P(churn) > 22%

### 8. Threshold Optimisation
Default threshold (0.5) gives balanced trade-off. Optimised F2 threshold (~0.285 for XGBoost) increases recall to ~93.9% at the cost of lower precision (~52.4%). Business should choose based on campaign budget and response rate estimates.

### 9. Missing Values Strategy
The modeling population has no missing values in predictor columns after the population filter. Imputers are included in the sklearn Pipeline as a defensive guardrail — primarily for internet-service conditional fields that are null when a customer has no internet service (e.g., `internet_type`, `avg_monthly_gb_download`).

| Column type | Imputer | Rationale |
|---|---|---|
| Numeric | `SimpleImputer(strategy="median")` | Robust to skewed distributions; does not pull the imputed value toward outliers |
| Categorical | `SimpleImputer(strategy="most_frequent")` | Preserves the dominant category; avoids introducing a spurious new level |
| Binary Yes/No | `.fillna(0)` at encoding time | Absence of service maps naturally to `No` (0) |

Imputers are **fit only on the training set** and applied to the test set — no leakage through imputation statistics.

### 10. Outlier Treatment
No explicit outlier removal or capping was applied. This was a deliberate decision:

1. **Tree models are outlier-robust** — XGBoost, LightGBM, and Random Forest split on thresholds, not distances. An extreme `total_charges` value only determines which side of a split a customer falls on; it does not distort the entire model the way it would in a linear model.
2. **Financial outliers carry real signal** — a customer with very high `total_charges` is a genuinely long-tenured, high-value customer. Capping or removing that observation would discard meaningful information.
3. **Logistic Regression** is the one model sensitive to scale; `StandardScaler` is applied in its pipeline, which reduces the leverage of extreme values without discarding them.

If a linear model were the champion, Winsorization (e.g., clip at 1st–99th percentile) would be recommended before scaling.

### 11. Normalization / Scaling Strategy
`StandardScaler` (zero mean, unit variance) is applied to all numeric columns inside the sklearn `ColumnTransformer`. However, its effect is **model-dependent**:

| Model | Scaling effect | Reason |
|---|---|---|
| Logistic Regression | **Material** — required for convergence and fair coefficient comparison | LR optimizes a linear combination of features; unscaled features with large magnitudes dominate the gradient |
| XGBoost / LightGBM / Random Forest | **None** — scaler is present in the pipeline but ignored in practice | Tree models split on ranked thresholds; multiplying a feature by a constant does not change the optimal split point |

The scaler is kept in all pipelines for consistency (single `ColumnTransformer` shared across models) and to future-proof against adding distance-based models (e.g., KNN, SVM) without pipeline changes.

### 12. Class Imbalance Strategy
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
| Threshold selection strategy | F2 (β=2) | Three strategies are implemented (`f2`, `youden`, `precision_recall`) but only F2 is used in training. F2 weights recall 2× over precision, reflecting the cost asymmetry: a missed churner costs $180 vs $4 for a wasted contact. `youden` (balances sensitivity + specificity equally) or `precision_recall` (maximises F1) would be appropriate if the business cost ratio changes | Re-run `find_optimal_threshold` with an alternative strategy if business cost estimates are revised |
| Risk tier — High cutoff | P(churn) ≥ 0.60 | Modelling convention, not derived from data or brief. Probability high enough to justify immediate, aggressive retention intervention (e.g. personal call). Business impact: determines the volume of highest-cost interventions per campaign cycle | Align with retention team capacity; if fewer than expected High-tier customers can be handled, raise the cutoff |
| Risk tier — Medium cutoff | 0.30 ≤ P(churn) < 0.60 | Lower bound chosen to align with the optimal F2 threshold (~0.29), capturing the uncertain zone just above the decision boundary. Business impact: Medium customers receive lower-cost interventions (e.g. email, SMS); if the team is saturated with High-tier contacts, Medium may be deprioritised | Validate empirically post-campaign: if Medium churn rate ≈ High churn rate, the boundary between them is not meaningful and cutoffs need revision |
| Risk tier — Low cutoff | P(churn) < 0.30 | Customers the model considers unlikely to churn; excluded from campaign by default. Business impact: any true churners in this band are missed — this is the irreducible cost of the model's false negative rate at the chosen threshold | Monitor actual churn rate in Low tier post-campaign; if materially above 0, consider lowering the Medium cutoff |

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
| `risk_tier` | High (≥0.6) / Medium (0.3–0.6) / Low (<0.3) — see note below |

> **Note — Risk Tier Thresholds**: the cutoffs 0.6 and 0.3 are **modelling assumptions**, not values derived from the data or specified in the assessment brief. They follow a common industry convention for three-band churn segmentation. The 0.3 lower bound was chosen to roughly align with the optimal F2 threshold (~0.29), so that "Medium" captures the uncertain zone above the decision boundary. In a production setting these cutoffs should be agreed with the business team based on: (1) retention team capacity — how many "High" cases can be handled per campaign cycle; (2) differentiated intervention costs per tier; and (3) empirical post-campaign validation — do "High" customers actually churn at a higher rate than "Medium"?

---

## Testing

```bash
uv run pytest tests/ -v
```

**Unit tests** are automated checks that verify each function in isolation does exactly what it is supposed to do. Unlike looking at model metrics (which tell you whether the model is good), unit tests verify that the *code* has no silent bugs — errors that would not crash the pipeline but would produce quietly incorrect results. For example: a bug in `add_derived_features` that computes `monthly_charges_ratio` from the wrong column would not raise an error, it would just produce lower AUC with no obvious explanation.

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
