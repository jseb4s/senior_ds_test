# Part B — written reasoning exercise

---

## B1

**Question:** Critically review the modeling strategy used in your Part A submission. Distinguish churn propensity from treatment effect. Propose a feasible pilot to estimate incremental retention and economic impact, including randomization or a justified observational alternative, statistical uncertainty and limitations.
*Guidance: Explain which conclusions cannot be obtained from the provided churn dataset alone.*

The Part A model predicts churn propensity — the probability a customer leaves given their current characteristics. This is not the same as the probability they respond to a retention offer. A customer with P(churn) = 0.90 who would churn regardless of a discount produces a true positive in the model but zero campaign value. The budget is wasted.

What cannot be concluded from this dataset alone: causal retention rate from any intervention, whether Fiber Optic customers would stay if offered a discount, or whether high-propensity customers are persuadable at all. The dataset is a single observational snapshot with no recorded interventions.

To estimate incremental retention, the only defensible approach is a randomised controlled trial. Score all eligible customers, stratify by risk decile, then randomly assign 70% to treatment (receive offer) and 30% to holdout (no contact). After 90 days, the difference in churn rates between groups is the incremental effect. A two-sided 95% CI on the estimate should be reported — not just the point estimate. If an RCT isn't feasible, propensity score matching on historical campaign data is an alternative, but selection bias cannot be fully ruled out and must be disclosed.

---

## B2

**Question:** Design logical data contracts for a reliable one-row-per-customer scoring dataset using the supplied SQLite tables. Explain grain, keys, leakage controls, checks and query performance. Describe how the design would change if timestamped event histories and large-scale processing were later added.
*Guidance: Do not invent timestamped events in the current data; focus on conceptual architecture and controls.*

The scoring table should have one row per customer per scoring date. The primary key is `(customer_id, scoring_date)` — no duplicates, no fan-out.

Hard leakage columns (`churn_category`, `churn_reason`) must be blocked at contract level, not just excluded in code. Cumulative billing fields (`total_charges`, `total_revenue`) are permitted for end-of-quarter batch scoring but must be flagged `availability_window: quarter_end` — for intra-quarter scoring they need to be replaced with rolling partial-period aggregates.

Minimum data quality checks enforced at pipeline entry: `customer_id` not null and unique, `tenure_in_months` between 0 and 500, `total_charges` non-negative, null rate on billing fields below 2%. If any check fails, the scoring run halts — it does not proceed with degraded data. Schema changes require a versioned contract update and re-validation before deployment.

If timestamped event histories were added, the grain would shift to `(customer_id, event_timestamp)` upstream, aggregated to `(customer_id, scoring_date)` in a summary table with an explicit `feature_as_of_date` to prevent point-in-time leakage. At scale, Delta Live Tables would handle incremental ETL, with time-windowed aggregations (e.g., `calls_last_30d`, `gb_downloaded_last_7d`) computed and stored as versioned summary tables before scoring.

---

## B3

**Question:** Propose a production-oriented design for your Part A solution. Cover separation of components, CI/CD, automated validation, model/data versioning, batch scoring, monitoring, rollback, ownership and prioritization.
*Guidance: Distinguish minimum viable controls from longer-term enhancements.*

The pipeline separates four concerns: data, training, registry, and scoring. Raw data lives in Delta Lake. The training job runs the full pipeline and logs to MLflow. The best run is promoted to the Model Registry under the `champion` alias after passing automated validation gates: AUC ≥ 0.88, recall ≥ 0.80, PSI < 0.2 versus the previous run. The batch scoring job loads the champion by alias and writes results to a Delta table. Marketing reads from that table — it never touches the model directly. Rollback is a tag change in the registry — no redeployment needed.

**Minimum viable controls (ship now):** MLflow experiment tracking, stratified train/test split, leakage column exclusion guard, unit tests, quarterly retraining trigger.

**Longer-term enhancements:** real-time scoring via Model Serving endpoint, champion/challenger A/B routing, fairness checks across demographic groups.

Monitoring tracks input drift via PSI, output score distribution shift, and lagged accuracy once churn labels arrive. A 5pp AUC drop triggers retraining and pauses the campaign.

---

## B4

**Question:** Hypothetical campaign: maximum 500 contacts, USD 4 cost per contact and USD 180 contribution margin per incremental retention. Define a targeting policy maximizing expected net contribution, explain when fewer than 500 contacts should be selected and how causal uncertainty should affect deployment decisions.
*Guidance: Do not use a churn probability as though it were an uplift estimate; quantify the break-even causal uplift.*

The economic floor is P(churn) > 22% — below that, expected benefit doesn't cover the $4 contact cost at a 10% assumed response rate. All model thresholds sit above this floor, so every flagged customer clears the break-even condition.

Fewer than 500 contacts should be selected when the marginal customer's expected net value is negative — that is, when P(churn) × 10% × $180 < $4. Contacting customers below this threshold loses money in expectation regardless of budget availability.

Without causal data, churn probability is the best available proxy for expected value. Contacts should be ordered by descending probability, capped at 500, and segmented by risk tier for differentiated interventions — personal call for High, email or SMS for Medium. Once the RCT from B1 runs, observed uplift per decile replaces probability as the ranking criterion.

---

## B5

**Question:** Propose governance controls for a recurring churn-scoring and marketing activation system, considering consent at activation time, PII minimization, access, lineage, auditability, fairness checks and stop/go decisions.
*Guidance: Define who should own escalation when an input validation or consent check fails.*

Three areas require explicit controls before production deployment.

On consent and PII minimisation: PII — personally identifiable information such as name, email, or phone number — should never enter the scoring pipeline. The model operates on `customer_id` only; identifying details are looked up at activation time by whoever sends the communication. Consent must be validated at scoring time, not at training time. If the consent check fails, affected customers are dropped. The default is stop, not proceed. Escalation path: Data Engineering flags the failure → Data Privacy Officer reviews → no activation proceeds without explicit sign-off.

On lineage and auditability: every scoring run logs model version, feature schema, and input data version in MLflow. Delta Lake time travel retains 90 days of score history, so any contact decision can be reproduced and audited against the exact model and data that generated it.

On fairness: before each production scoring run, check score distribution by age group and gender. If the gap between demographic groups exceeds 10 percentage points relative to the overall churn rate, halt deployment and escalate to the Head of Data and Legal before proceeding. If input validation fails (null rate above 2%, unexpected status values), the scoring run halts and Data Engineering Lead is notified immediately.
