"""
Feature engineering pipeline.

Design decisions:
- Uses sklearn Pipeline/ColumnTransformer for reproducibility and to prevent
  train/test leakage in transformations (e.g., imputation statistics fit only on train).
- Categorical encoding: OrdinalEncoder inside pipeline (compatible with tree models).
  For LR, OneHotEncoder is used instead (injected via config).
- Binary Yes/No columns: mapped to 0/1 before pipeline to reduce cardinality.
- Derived features added before pipeline to maximise interpretability.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)

BINARY_YES_NO = [
    "phone_service",
    "multiple_lines",
    "internet_service",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
    "unlimited_data",
    "married",
]

TARGET = "churn_label"
LEAKAGE_COLS = ["churn_category", "churn_reason", "customer_status"]
ID_COLS = ["customer_id", "zip_code"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven feature engineering.

    - monthly_charges_ratio: smoothed monthly spend (total_charges / tenure).
      Customers with high spend-per-month but short tenure are higher churn risk.
    - net_revenue_per_month: revenue net of refunds per month.
    - has_extra_data_charges: boolean — exceeded data plan at least once.
    - streaming_count: number of streaming services subscribed. Bundle stickiness proxy.
    - is_fiber: fiber optic customers churn more (competitor sensitivity).
    - is_month_to_month: proxy contract commitment from payment/tenure patterns.
      Note: no contract_type field in this dataset, so we use tenure < 12 as a proxy.
    """
    df = df.copy()

    df["monthly_charges_ratio"] = np.where(
        df["tenure_in_months"] > 0,
        df["total_charges"] / df["tenure_in_months"],
        0.0,
    )

    df["net_revenue_per_month"] = np.where(
        df["tenure_in_months"] > 0,
        (df["total_revenue"]) / df["tenure_in_months"],
        0.0,
    )

    df["has_extra_data_charges"] = (df["total_extra_data_charges"] > 0).astype(int)

    streaming_cols = ["streaming_tv", "streaming_movies", "streaming_music"]
    df["streaming_count"] = df[streaming_cols].apply(
        lambda row: sum(str(v).strip().lower() == "yes" for v in row), axis=1
    )

    df["is_fiber"] = (
        df["internet_type"].str.strip().str.lower() == "fiber optic"
    ).astype(int)

    df["is_short_tenure"] = (df["tenure_in_months"] < 12).astype(int)

    df["referrals_zero"] = (df["number_of_referrals"] == 0).astype(int)

    df["revenue_to_charges_ratio"] = np.where(
        df["total_charges"] > 0,
        df["total_revenue"] / df["total_charges"],
        1.0,
    )

    logger.info("Derived features added: %d total columns", df.shape[1])
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No and gender to 0/1 — avoids OHE explosion for boolean fields."""
    df = df.copy()
    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)

    return df


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    use_ohe: bool = False,
) -> ColumnTransformer:
    """
    Returns a ColumnTransformer.
    use_ohe=True for linear models; OrdinalEncoder for tree-based models.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    if use_ohe:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    else:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features(
    df: pd.DataFrame,
    exclude_cols: Optional[list] = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Returns (X, y, numeric_cols, categorical_cols).

    Caller is responsible for splitting X into train/test before fitting a preprocessor
    to avoid any form of data leakage through imputation or scaling statistics.
    """
    if exclude_cols is None:
        exclude_cols = []

    df = add_derived_features(df)
    df = encode_binary_columns(df)

    drop_cols = set(LEAKAGE_COLS + ID_COLS + [TARGET]) | set(exclude_cols)
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = df[TARGET]

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(
        "Features prepared — numeric: %d, categorical: %d",
        len(numeric_cols),
        len(categorical_cols),
    )

    return feature_df, y, numeric_cols, categorical_cols
