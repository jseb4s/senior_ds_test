"""
Data loading and validation layer.

Responsible for:
- Reading from the SQLite source (read-only)
- Applying modeling population filter (Churned / Stayed only)
- Creating the binary churn label
- Surfacing data quality checks
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_raw(db_path: str | Path) -> pd.DataFrame:
    """
    Join all modeling-relevant tables. The modeling_dataset view is a convenience
    starting point but we also pull churn_category / churn_reason for EDA reference
    (they are excluded from predictors at feature-engineering time).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    query = """
    SELECT
        c.customer_id,
        c.gender,
        c.age,
        c.married,
        c.zip_code,
        c.number_of_referrals,
        c.tenure_in_months,
        sf.phone_service,
        sf.avg_monthly_long_distance_charges,
        sf.multiple_lines,
        sf.internet_service,
        sf.internet_type,
        sf.avg_monthly_gb_download,
        sf.streaming_tv,
        sf.streaming_movies,
        sf.streaming_music,
        sf.unlimited_data,
        bf.payment_method,
        bf.total_charges,
        bf.total_refunds,
        bf.total_extra_data_charges,
        bf.total_long_distance_charges,
        bf.total_revenue,
        zp.population AS zip_population,
        co.customer_status,
        co.churn_category,
        co.churn_reason
    FROM customers c
    JOIN service_features sf   ON c.customer_id = sf.customer_id
    JOIN billing_features bf   ON c.customer_id = bf.customer_id
    JOIN churn_outcomes co     ON c.customer_id = co.customer_id
    LEFT JOIN zip_population zp ON c.zip_code = zp.zip_code
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    logger.info("Raw dataset loaded: %d rows × %d cols", *df.shape)
    return df


def _run_quality_checks(df: pd.DataFrame) -> None:
    """Fail-fast checks before training."""
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.5]
    if not high_null.empty:
        logger.warning("High null columns (>50%%): %s", high_null.to_dict())

    if df["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id detected — data integrity issue.")

    expected_statuses = {"Churned", "Stayed", "Joined"}
    actual = set(df["customer_status"].unique())
    unexpected = actual - expected_statuses
    if unexpected:
        raise ValueError(f"Unexpected customer_status values: {unexpected}")


def load_modeling_dataset(
    db_path: str | Path,
    modeling_population: Optional[list] = None,
) -> pd.DataFrame:
    """
    Returns a clean DataFrame ready for feature engineering.

    Population choice rationale:
    - 'Joined' customers are excluded because they were acquired this quarter and
      have not had sufficient tenure to exhibit churn behavior. Including them would
      bias the model toward predicting non-churn for new customers — which is trivially
      true but unactionable from a retention campaign standpoint.
    - 'Stayed' vs 'Churned' gives a clean binary outcome on the stable customer base.
    """
    if modeling_population is None:
        modeling_population = ["Churned", "Stayed"]

    df = _load_raw(db_path)
    _run_quality_checks(df)

    before = len(df)
    df = df[df["customer_status"].isin(modeling_population)].copy()
    logger.info(
        "Population filter applied (%s): %d → %d rows",
        modeling_population,
        before,
        len(df),
    )

    df["churn_label"] = (df["customer_status"] == "Churned").astype(int)
    churn_rate = df["churn_label"].mean()
    logger.info(
        "Churn rate in modeling population: %.1f%% (%d churned / %d total)",
        churn_rate * 100,
        df["churn_label"].sum(),
        len(df),
    )
    return df
