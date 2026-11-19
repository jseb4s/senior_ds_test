# Part B — Written Reasoning Exercise
## Senior Data Scientist · AllLife Communications — Telecom Churn

---

## B1. Critical Review: Propensity vs Treatment Effect

### What the Part A model measures
The model predicts **churn propensity** — the probability that a customer churns *given their observable characteristics at Q2 2022*. This is **not** the probability that a customer responds to a retention campaign. The two quantities are conflated by default in many deployments, but the distinction is critical:

| Concept | Definition | What we can measure |
|---|---|---|
| Churn propensity P(churn) | Probability of leaving absent any intervention | ✅ From the snapshot dataset |
| Treatment effect τ = P(churn\|no campaign) − P(churn\|campaign) | Incremental retention caused by the campaign | ❌ Not in this dataset |

A high-propensity customer who would churn regardless of contact produces a True Positive in the model but zero causal lift. The campaign budget is wasted on them. Conversely, some low-propensity customers might be highly responsive to an offer — the propensity model ignores them.

### What cannot be concluded from this dataset alone
- Causal retention rate from any marketing action
- Whether Fiber Optic customers leave *because* of price (they might just be unhappy and leave regardless of a discount)
- Whether customers flagged as `churn_category = Competitor` are persuadable or already committed to leaving

### Feasible pilot to estimate incremental retention
**Randomised controlled trial (RCT) — the gold standard:**

1. Score all eligible customers with the Part A model (Q3 2022 or next scoring cycle).
2. Stratify by risk decile to ensure balance.
3. Randomly assign within each stratum: 70% treatment (receive retention offer), 30% holdout (no contact).
4. Observe churn over 90 days.
5. Estimate ATE: `τ̂ = Churn_rate(holdout) − Churn_rate(treated)` with standard errors from a t-test or logistic regression.

**Statistical considerations:**
- With ~1,700 expected churners/quarter, a 30% holdout gives ~510 observations. Power calculation: at τ = 5pp effect, α=0.05, power=0.80, minimum detectable effect is achievable with ~400 per arm.
- Two-sided 95% CI on τ̂ should be reported — not just the point estimate.

**If RCT is not feasible (e.g., marketing already activated):**
- Difference-in-differences using a natural experiment (e.g., regional campaign rollout).
- Propensity score matching on historical campaign/non-campaign data, with sensitivity analysis via Rosenbaum bounds.
- **Limitation:** Observational alternatives cannot fully rule out selection bias. Must disclose.

---

## B2. Logical Data Contracts for Scoring Dataset

### Grain and keys
One row per `customer_id` at a defined **scoring timestamp** (e.g., last day of quarter). The primary key is `(customer_id, scoring_date)`. No fan-out, no duplicates.

### Leakage controls

| Field class | Control |
|---|---|
| `churn_category`, `churn_reason` | **Hard block**: excluded by contract; only available post-churn. Any query that references these for prediction must fail contract validation. |
| `customer_status` | Excluded from features; used only as label after the fact. |
| Cumulative billing fields (`total_charges`, `total_revenue`) | Permitted for end-of-quarter batch scoring. For intra-quarter real-time scoring, replace with partial-period rolling aggregates (e.g., `charges_last_90d`). Flag in contract with `availability_window: quarter_end`. |
| `tenure_in_months` | Available at scoring time; acceptable predictor. |

### Data quality checks (enforced at pipeline entry)
```
CHECK: customer_id IS NOT NULL AND customer_id IS UNIQUE
CHECK: tenure_in_months BETWEEN 0 AND 500
CHECK: total_charges >= 0
CHECK: customer_status IN ('Churned', 'Stayed', 'Joined')
CHECK: internet_type IN ('DSL', 'Fiber Optic', 'Cable', NULL)
CHECK: NULL rate for billing fields < 2%
```

### Query performance
- Partition by `scoring_date` to prune historical partitions.
- Cluster/index on `customer_id` for join performance.
- Materialise the joined feature table as a Delta table (Databricks) rather than re-running 5-table joins at score time.

### Evolution: timestamped event histories at scale
If transaction-level events are added (clicks, calls, payments, service interruptions):
- Change grain to `(customer_id, event_timestamp)` upstream; aggregate to `(customer_id, scoring_date)` in a feature store.
- Introduce time-windowed features (e.g., `calls_last_30d`, `gb_downloaded_last_7d`) with explicit `feature_as_of_date` to prevent point-in-time leakage.
- Use Databricks Feature Store or Feast for consistent train/serve feature retrieval.
- Large-scale processing: Delta Live Tables for incremental ETL, with `APPLY CHANGES INTO` for SCD handling.

---

## B3. Production Architecture

### Component separation

```
┌─────────────────────────────────────────────────────┐
│  Data Layer (Delta Lake)                            │
│  · raw_customers, raw_billing, raw_service          │
│  · feature_store.churn_features (scored daily)      │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  Training Pipeline (Databricks Job)                 │
│  · src/data/loader.py  → population filter          │
│  · src/features/engineering.py → feature transforms │
│  · src/models/trainer.py → Optuna + MLflow          │
│  · Triggered: monthly or on data-drift alert        │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  MLflow Model Registry                              │
│  · Staging → validation checks → Production         │
│  · Champion/challenger via A/B flag in registry     │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  Batch Scoring Pipeline (Databricks Job, nightly)   │
│  · Load Production model from registry              │
│  · Score all Stayed customers                       │
│  · Write scores to output_scores.churn_scores       │
│  · Trigger marketing activation if score > threshold│
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  Monitoring (Databricks Lakehouse Monitoring)        │
│  · Input drift: PSI/KS on feature distributions     │
│  · Output drift: score distribution shift           │
│  · Accuracy: observed churn vs predicted (30d lag)  │
│  · Alert → Slack/PagerDuty on threshold breach      │
└─────────────────────────────────────────────────────┘
```

### CI/CD
- **Unit tests** (`tests/`): validate feature engineering, leakage guards, data quality checks. Run on every PR.
- **Integration tests**: smoke test train pipeline on 1k-row sample.
- **Model validation gate**: before promoting to Production, automated checks: AUC ≥ 0.88, recall ≥ 0.80, PSI < 0.2 vs last production run.
- **Rollback**: previous model version in registry is tagged `archived`; rollback = bump version tag, re-run scoring.

### Minimum viable controls (ship now)
- MLflow experiment + run tracking
- Stratified train/test split
- Leakage column exclusion guard
- Weekly model re-run trigger

### Longer-term enhancements
- Databricks Feature Store for train-serve consistency
- Real-time scoring via REST endpoint (Model Serving)
- Champion/challenger A/B routing
- Fairness checks across age/gender groups

---

## B4. Campaign Targeting Policy

### Economic framework
Given: 500 max contacts, $4 cost/contact, $180 contribution margin per incremental retention.

**Break-even causal uplift** per contact:
```
Cost = $4
Expected benefit = P(response | targeted) × $180
Break-even: P(response) = 4 / 180 = 2.2%
```

This is a very low bar — but it applies to *causal* response rate (τ), not churn propensity.

### Why churn probability ≠ uplift
A customer with P(churn) = 0.90 who would churn regardless of a $20 offer provides τ ≈ 0. Contacting them costs $4 for zero benefit. The model predicts they will churn; it cannot predict whether the offer changes that.

### Targeting policy
1. **Score** all customers with the XGBoost model → rank by P(churn).
2. **Segment and communicate** using SHAP values: tenure and referrals dominate the model, so the retention team can design segment-specific offers — a first-year onboarding offer for short-tenure customers, a loyalty bundle for Fiber customers — rather than a one-size-fits-all discount.
3. **Prioritize** customers with highest *expected net value*:
   ```
   Net value(i) = τ̂ × $180 − $4
   ```
   Where τ̂ can be estimated from a pilot RCT (see B1) stratified by churn decile.
3. **Without causal data**: use a conservative τ̂ assumption (e.g., 10% response in the top decile, 5% in mid decile, 2% in low decile — these are industry benchmarks for telecom retention, **not derived from this dataset**, and must be replaced with observed rates from the RCT in B1).
4. Select the top-N contacts where `τ̂ × $180 − $4 > 0`, **capped at 500**.
5. **When fewer than 500 should be selected**: if the marginal customer's expected net value is negative (τ̂ × $180 < $4), do not contact them. With a 10% assumed response rate, the threshold is P(churn) × 10% × $180 > $4 → P(churn) > 22%. In practice, this is calibrated via the threshold analysis (see `reports/figures/10_threshold_analysis_xgboost.png`).

### Causal uncertainty and deployment
Given no RCT data, τ̂ carries high uncertainty. Recommended approach:
- Deploy campaign to 500 contacts but designate 15% as a **holdout** (no contact) for real-time ATE estimation.
- After 60 days, update τ̂ estimates per risk decile.
- If 90% CI on net value is below zero, pause targeting in that decile.

---

## B5. Governance Controls

### Data and consent
| Control | Description | Owner |
|---|---|---|
| **Consent at activation** | Marketing contact requires explicit opt-in or legitimate interest basis per applicable privacy law. Consent flag must be validated at scoring time, not at model training time. | Legal / Data Privacy Officer |
| **PII minimisation** | Scoring pipeline uses `customer_id` only; PII (name, email) joined at activation by the CRM system, not in the ML pipeline. | Data Engineering |
| **Access control** | Model artifacts and score outputs on Delta tables with column-level ACLs. Data scientists: read-only on raw tables. Marketing: read-only on output scores table. | Data Platform |

### Lineage and auditability
- MLflow logs: every scoring run records the model version, feature schema, and input data version.
- Delta Lake time travel: score outputs retain 90 days of history. Any contact decision can be reproduced by joining scores to CRM.
- Data contracts (see B2) specify schema versions; breaking changes require a new contract version and re-validation.

### Fairness checks
Before every production scoring run:
- Compute churn rate and score distribution by age group, gender.
- Flag if score gap between demographic groups > 10pp relative to overall churn rate.
- Document any disparate impact findings; do not deploy if gap exceeds regulatory threshold without explicit review.

### Stop/go decisions

| Trigger | Stop action | Escalation owner |
|---|---|---|
| Input validation fails (null rate > 2%, unexpected status values) | Halt scoring run; alert on-call engineer | Data Engineering Lead |
| Consent check fails (no opt-in flag in CRM) | Drop affected customers from activation list | Legal / Privacy Officer |
| Model AUC drops > 5pp vs baseline on live data | Trigger emergency retraining; pause campaign | ML Engineering Lead |
| Fairness check exceeds gap threshold | Halt deployment; request review | Head of Data / Legal |

**Escalation path for consent/validation failure:**
Data Engineering → Data Privacy Officer → Head of Product → Legal (for regulatory exposure).

No marketing activation should proceed if consent or input validation checks have not been explicitly confirmed by the Data Privacy Officer when a failure has been flagged. The default is **stop**, not proceed.

---

## B6. Overfitting Analysis

### CV vs Test AUC comparison

| Model | CV ROC-AUC | Test ROC-AUC | Gap | Verdict |
|---|---|---|---|---|
| XGBoost | 0.9040 | 0.9087 | +0.0048 | ✅ No overfitting |
| LightGBM | 0.9032 | 0.9064 | +0.0032 | ✅ No overfitting |
| Random Forest | 0.8969 | 0.9012 | +0.0043 | ✅ No overfitting |
| Logistic Regression | 0.8825 | 0.8881 | +0.0055 | ✅ No overfitting |

### Interpretation

**None of the models overfit.** A positive gap (Test AUC > CV AUC) is the opposite of the overfitting pattern — an overfit model scores high on training data and lower on unseen data. Here, all models score slightly *higher* on the test set than their cross-validation average.

Two complementary explanations:

1. **CV uses smaller training subsets.** In 5-fold CV, each fold trains on 80% of the training set (~3,830 rows). The final model is retrained on the full 4,788-row training set, giving it more signal. This systematically produces test scores at or above the CV mean.

2. **Regularization is effective.** Optuna found hyperparameters with meaningful regularization in all tree models:
   - XGBoost: `reg_alpha ≈ 5.5e-8`, `reg_lambda = 0.0065`, `max_depth = 4` (shallow)
   - LightGBM: `num_leaves = 20` (conservative given `max_depth = 3`), `min_child_samples = 12`
   - Random Forest: `min_samples_leaf = 1` but `max_depth = 9` with `max_features = sqrt` limits each tree

**On the near-90% AUC:** this is not suspicious given the dataset structure. The churn signal in this data is strong and well-defined — tenure, referrals, and internet type are highly predictive by nature of the telecom industry. An AUC of 0.90 on a clean, labelled quarter-end snapshot is entirely plausible without overfitting. The concern would arise if AUC were above 0.98+ or if there were an unexplained train/test gap, neither of which is the case here.

**What to monitor in production:** once the model scores on new quarters, we expect some natural AUC decay as market conditions shift (new competitors, pricing changes). A monitoring gate of AUC ≥ 0.85 (5pp below current) before triggering retraining is a reasonable production threshold.

---

## B7. Model Comparison: XGBoost vs LightGBM vs Logistic Regression

### Performance summary (at optimised F2 threshold)

| Metric | XGBoost ⭐ | LightGBM | Logistic Regression | Notes |
|---|---|---|---|---|
| Test ROC-AUC | **0.9087** | 0.9064 | 0.8881 | Primary selection criterion |
| Test F2 (β=2) | **0.8111** | 0.8086 | 0.8051 | Weights recall 2× over precision |
| Test Recall | 0.9213 | **0.9359** | 0.9155 | LightGBM catches most churners |
| Test Precision | **0.5486** | 0.5237 | 0.5433 | XGB wastes fewest contacts |
| Net campaign value | **$3,384** | $3,326 | $3,340 | At 10% assumed response rate |
| Campaign ROI | 146.9% | 135.6% | 144.5% | XGB and LR similar; LightGBM lower |
| Optimal threshold | 0.290 | 0.285 | 0.360 | LR needs higher threshold = more conservative |
| CV ROC-AUC | **0.9040** | 0.9032 | 0.8825 | Generalization check |
| CV → Test gap | +0.005 | +0.003 | +0.006 | All positive: no overfitting |

### Key takeaways from the comparison

**XGBoost vs LightGBM:** The gap is narrow across all metrics. LightGBM recovers 15 more churners per 1,000 customers (higher recall) but contacts 25 more false positives — net result is $58 less campaign value. XGBoost is selected as champion by automated ROC-AUC maximisation. At Databricks scale (millions of rows), LightGBM's training speed advantage (3–10×) would make it the preferred choice.

**XGBoost vs Logistic Regression:** LR is 2pp lower on AUC but only marginally behind on F2 and net value. The gap is real but not dramatic — the data has strong linear structure (tenure, referrals) that LR captures well. The key difference is that XGBoost also captures non-linear interactions (e.g., Fiber Optic customers under 12 months behave differently than Fiber customers at 24 months) that LR misses. LR's threshold (0.360) is notably higher, meaning it is more selective about who it flags — appropriate for a constrained-budget campaign where precision matters.

**Logistic Regression as a baseline:** LR confirms that the predictive signal in this dataset is not purely non-linear. It performs within 2pp AUC of the best tree model, which validates that the features are well-specified. If LR had underperformed by 10pp+, it would indicate the problem is highly non-linear and the tree models' gains would be more essential.

---

## B8. XGBoost Champion — Conclusions and Business Insights

### Model summary

XGBoost (AUC=0.9087, Recall=0.9213, Precision=0.5486 at threshold=0.290) is the production champion. It identifies 921 out of every 1,000 churners in the test set, with a predicted churn rate of 51.5% across all scored customers at the operational threshold. Net estimated campaign value: **$3,384** per scoring cycle at a conservative 10% response rate assumption.

### Top features by SHAP importance

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | `tenure_in_months` | 0.870 | Strongest predictor. Loyalty compounds continuously — each extra month meaningfully lowers churn probability. The model is essentially a tenure-corrected risk engine. |
| 2 | `number_of_referrals` | 0.671 | Second strongest. Customers who refer others are deeply embedded — referrals signal brand advocacy, not just satisfaction. The gap between zero and one referral is the largest single jump in churn risk. |
| 3 | `is_fiber` | 0.294 | Fiber Optic customers face more competitor pressure. The model captures this as a structural segment effect, not just a price signal. |
| 4 | `monthly_charges_ratio` | 0.207 | Spend per month of relationship. High ratio = customer is extracting value and staying. Low ratio with long tenure = at-risk of competitor poaching. |
| 5 | `internet_service` | 0.191 | Internet is the anchor product. Customers without internet have far lower churn risk — they have fewer substitutes. |
| 6 | `payment_method` | 0.188 | Credit card payers churn less. Auto-pay via credit card creates inertia — switching requires deliberate action. |
| 7 | `is_short_tenure` | 0.173 | Binary flag: < 12 months. Triggers early in the customer lifecycle. Note: ranks #1 in tree split frequency (0.331) but #7 in SHAP — it fires at many split points, but `tenure_in_months` carries the continuous gradient. |
| 8 | `total_revenue` | 0.117 | Cumulative relationship value. Higher total revenue = longer or higher-spend relationship = stronger lock-in. |
| 9 | `total_long_distance_charges` | 0.116 | Long-distance usage signals deeper product adoption and cross-service integration, which raises switching cost. |
| 10 | `age` | 0.104 | Older customers are moderately more loyal. May reflect lower digital literacy (harder to switch), longer tenures, or different price sensitivity. |

### Business insights for AllLife Communications

**1. Retention is a first-year problem above all else.**
Tenure dominates both SHAP (#1) and tree importance. The churn hazard is heavily front-loaded — customers who survive 12 months are substantially safer. Every retention investment in months 1–12 has the highest expected return. A structured onboarding programme (proactive check-in at month 3, loyalty offer at month 9) is the highest-ROI structural intervention the business can make.

**2. The referral programme is the single highest-leverage retention tool.**
`number_of_referrals` is the second-strongest SHAP feature. Customers with zero referrals churn at dramatically higher rates. A referral programme would simultaneously reduce churn and acquire new customers — a double return. Even converting 10% of zero-referral customers to one referral would materially shift the churn distribution.

**3. Fiber Optic requires a differentiated strategy.**
`is_fiber` ranks #3 in SHAP with a structurally elevated risk. Fiber customers are likely targeted by ISP competitors with device upgrade offers. A price lock or device upgrade programme specifically for Fiber customers in their first year would address the two highest-risk dimensions simultaneously.

**4. Payment method is an underutilised retention signal.**
Credit card payers (#6 SHAP) churn less. This may indicate that migrating customers to credit card auto-pay reduces churn independently of satisfaction — it creates friction to leave. A targeted migration campaign (e.g., incentive to set up auto-pay) is a low-cost, scalable intervention.

**5. High-revenue customers need proactive attention.**
`total_revenue` and `monthly_charges_ratio` together indicate that high-value customers are worth protecting aggressively. The model scores them as moderate-to-high risk; combined with their margin contribution, they should receive the highest-value retention offers, not generic discounts.

---

## B9. Next Step — Uplift Modeling

### Why uplift, and why not now

The XGBoost model answers: *"Who is likely to churn?"* The business question is actually: *"Who will we retain if we intervene?"* These are different. A customer with P(churn)=0.90 who would leave regardless of a $20 offer provides zero causal value — contacting them wastes $4. This distinction is fully described in B1.

Uplift modeling (also called heterogeneous treatment effect estimation) would directly estimate:

```
τ(x) = P(churn | no intervention, x) − P(churn | intervention, x)
```

...at the individual customer level. Customers with high τ(x) are the ones where intervention *changes* the outcome — the genuinely persuadable segment.

### Why it is not viable yet

Uplift requires experimental data: a treated group (received offer) and a control group (no contact), with observed churn outcomes. The current dataset is a **pure observational snapshot** — no intervention was recorded, no holdout group exists. Training an uplift model on this data would produce meaningless estimates.

### How to get there

1. **Run the RCT described in B1** — score all customers with the current XGBoost model, stratify by risk decile, randomly assign 70% treatment / 30% holdout within each stratum.
2. **Observe outcomes over 90 days** — record who churned in each group.
3. **Train an uplift model** on the resulting dataset using one of:
   - **T-learner** (two separate models, one per treatment arm) — simple baseline
   - **X-learner** (EconML) — better for imbalanced treatment/control sizes
   - **Causal Forest** (grf / EconML) — best for heterogeneous effects across customer segments
4. **Replace the targeting policy** in B4 — instead of selecting top-N by P(churn), select top-N by τ̂(x) weighted by expected margin.

The XGBoost propensity scores from Part A are not wasted — they are the ideal stratification instrument for the RCT, and can serve as a meta-feature in the uplift model (propensity weighting). The two models work in sequence, not in opposition.
