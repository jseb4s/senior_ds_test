"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.features.engineering import (
    add_derived_features,
    encode_binary_columns,
    prepare_features,
    LEAKAGE_COLS,
)


def _make_df(n=10):
    """Minimal synthetic DataFrame that mimics the SQLite schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.integers(18, 80, n),
        "married": rng.choice(["Yes", "No"], n),
        "zip_code": rng.integers(10000, 99999, n),
        "number_of_referrals": rng.integers(0, 10, n),
        "tenure_in_months": rng.integers(1, 72, n),
        "phone_service": rng.choice(["Yes", "No"], n),
        "multiple_lines": rng.choice(["Yes", "No"], n),
        "internet_service": rng.choice(["Yes", "No"], n),
        "internet_type": rng.choice(["Fiber Optic", "DSL", "Cable"], n),
        "avg_monthly_gb_download": rng.uniform(1, 100, n),
        "avg_monthly_long_distance_charges": rng.uniform(0, 50, n),
        "streaming_tv": rng.choice(["Yes", "No"], n),
        "streaming_movies": rng.choice(["Yes", "No"], n),
        "streaming_music": rng.choice(["Yes", "No"], n),
        "unlimited_data": rng.choice(["Yes", "No"], n),
        "payment_method": rng.choice(["Credit Card", "Bank Transfer", "Mailed Check"], n),
        "total_charges": rng.uniform(100, 5000, n),
        "total_refunds": rng.uniform(0, 50, n),
        "total_extra_data_charges": rng.uniform(0, 30, n),
        "total_long_distance_charges": rng.uniform(0, 200, n),
        "total_revenue": rng.uniform(100, 5000, n),
        "zip_population": rng.integers(1000, 100000, n),
        "customer_status": rng.choice(["Churned", "Stayed"], n),
        "churn_category": rng.choice(["Competitor", "Price", None], n),
        "churn_reason": rng.choice(["Better device offers", "Price too high", None], n),
        "churn_label": rng.integers(0, 2, n),
    })


class TestAddDerivedFeatures:
    def test_adds_expected_columns(self):
        df = _make_df()
        result = add_derived_features(df)
        expected = [
            "monthly_charges_ratio",
            "net_revenue_per_month",
            "has_extra_data_charges",
            "streaming_count",
            "is_fiber",
            "is_short_tenure",
            "referrals_zero",
            "revenue_to_charges_ratio",
        ]
        for col in expected:
            assert col in result.columns, f"Missing derived column: {col}"

    def test_no_mutation_of_input(self):
        df = _make_df()
        cols_before = set(df.columns)
        add_derived_features(df)
        assert set(df.columns) == cols_before

    def test_monthly_charges_ratio_zero_tenure(self):
        df = _make_df(5)
        df["tenure_in_months"] = 0
        result = add_derived_features(df)
        assert (result["monthly_charges_ratio"] == 0).all()

    def test_streaming_count_range(self):
        df = _make_df(20)
        result = add_derived_features(df)
        assert result["streaming_count"].between(0, 3).all()

    def test_is_fiber_binary(self):
        df = _make_df()
        result = add_derived_features(df)
        assert set(result["is_fiber"].unique()).issubset({0, 1})

    def test_has_extra_data_charges_binary(self):
        df = _make_df()
        result = add_derived_features(df)
        assert set(result["has_extra_data_charges"].unique()).issubset({0, 1})


class TestEncodeBinaryColumns:
    def test_yes_no_mapped_to_int(self):
        df = _make_df()
        result = encode_binary_columns(df)
        for col in ["phone_service", "multiple_lines", "internet_service",
                    "streaming_tv", "streaming_movies", "streaming_music",
                    "unlimited_data", "married"]:
            if col in result.columns:
                assert result[col].dtype in [np.int32, np.int64, int]
                assert set(result[col].unique()).issubset({0, 1})

    def test_gender_encoded(self):
        df = _make_df()
        result = encode_binary_columns(df)
        assert set(result["gender"].unique()).issubset({0, 1})

    def test_no_mutation_of_input(self):
        df = _make_df()
        original_values = df["phone_service"].copy()
        encode_binary_columns(df)
        pd.testing.assert_series_equal(df["phone_service"], original_values)


class TestPrepareFeatures:
    def test_returns_four_outputs(self):
        df = _make_df(50)
        X, y, num_cols, cat_cols = prepare_features(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(num_cols, list)
        assert isinstance(cat_cols, list)

    def test_no_leakage_columns_in_X(self):
        df = _make_df(50)
        X, _, _, _ = prepare_features(df)
        for col in LEAKAGE_COLS:
            assert col not in X.columns, f"Leakage column in X: {col}"

    def test_target_not_in_X(self):
        df = _make_df(50)
        X, _, _, _ = prepare_features(df)
        assert "churn_label" not in X.columns

    def test_exclude_cols_respected(self):
        df = _make_df(50)
        X, _, _, _ = prepare_features(df, exclude_cols=["payment_method"])
        assert "payment_method" not in X.columns

    def test_y_binary(self):
        df = _make_df(50)
        _, y, _, _ = prepare_features(df)
        assert set(y.unique()).issubset({0, 1})

    def test_numeric_categorical_partition(self):
        df = _make_df(50)
        X, _, num_cols, cat_cols = prepare_features(df)
        assert set(num_cols + cat_cols) == set(X.columns)
