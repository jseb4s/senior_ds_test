"""
Exploratory Data Analysis module.

Produces:
- Univariate statistics
- Churn rate by feature segment
- Correlation analysis
- Distribution plots saved to reports/figures/
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"Churned": "#e74c3c", "Stayed": "#2ecc71"}


def _save(fig, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def describe_dataset(df: pd.DataFrame) -> None:
    logger.info("Dataset shape: %d rows × %d cols", *df.shape)

    status_counts = df["customer_status"].value_counts()
    logger.info("customer_status distribution:\n%s", status_counts.to_string())

    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        logger.info("No missing values detected.")
    else:
        logger.info("Missing values (%%):\n%s", null_pct.to_string())

    num_cols = df.select_dtypes(include="number").columns.tolist()
    logger.info("Numeric summary:\n%s", df[num_cols].describe().to_string())

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        logger.info("  %s value counts:\n%s", col, df[col].value_counts().to_string())


# ---------------------------------------------------------------------------
# Churn rate analysis
# ---------------------------------------------------------------------------

def churn_rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute churn rate per category of a column."""
    summary = (
        df.groupby(col)["churn_label"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
        .sort_values("churn_rate", ascending=False)
    )
    return summary


def plot_churn_rate_bars(df: pd.DataFrame, columns: list, ncols: int = 3) -> None:
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        summary = churn_rate_by_category(df, col)
        ax = axes[i]
        bars = ax.barh(
            summary.index.astype(str),
            summary["churn_rate"] * 100,
            color="#e74c3c",
            alpha=0.8,
        )
        ax.set_xlabel("Churn Rate (%)")
        ax.set_title(col.replace("_", " ").title())
        for bar, (_, row) in zip(bars, summary.iterrows()):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"n={row['n']:,}", va="center", fontsize=8,
            )
        ax.axvline(df["churn_label"].mean() * 100, color="grey", linestyle="--", linewidth=1)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Churn Rate by Feature Category", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "churn_rate_by_category")


def plot_numeric_distributions(df: pd.DataFrame, columns: list, ncols: int = 3) -> None:
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        for status in ["Stayed", "Churned"]:
            subset = df[df["customer_status"] == status][col].dropna()
            ax.hist(subset, bins=30, alpha=0.5, label=status, color=PALETTE[status])
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions (Churned vs Stayed)", fontsize=14)
    fig.tight_layout()
    _save(fig, "numeric_distributions")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include="number").drop(columns=["churn_label"], errors="ignore")
    corr = num_df.corrwith(df["churn_label"]).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, len(corr) * 0.35)))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in corr]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Churn Label")
    ax.set_title("Feature Correlation with Churn")
    fig.tight_layout()
    _save(fig, "feature_correlation_with_churn")


def plot_churn_by_tenure(df: pd.DataFrame) -> None:
    df_copy = df.copy()
    df_copy["tenure_bin"] = pd.cut(
        df_copy["tenure_in_months"],
        bins=[0, 6, 12, 24, 36, 60, np.inf],
        labels=["0-6", "7-12", "13-24", "25-36", "37-60", "60+"],
    )
    summary = churn_rate_by_category(df_copy, "tenure_bin")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary.index.astype(str), summary["churn_rate"] * 100, color="#e74c3c", alpha=0.8)
    ax.axhline(df["churn_label"].mean() * 100, color="grey", linestyle="--", label="Overall churn rate")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Tenure Band")
    ax.legend()
    fig.tight_layout()
    _save(fig, "churn_by_tenure")


def plot_revenue_vs_churn(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col in zip(axes, ["total_revenue", "tenure_in_months"]):
        for status in ["Stayed", "Churned"]:
            subset = df[df["customer_status"] == status][col].dropna()
            ax.hist(subset, bins=40, alpha=0.6, label=status, density=True, color=PALETTE[status])
        ax.set_title(col.replace("_", " ").title())
        ax.legend()

    fig.suptitle("Revenue and Tenure by Churn Status", fontsize=13)
    fig.tight_layout()
    _save(fig, "revenue_tenure_vs_churn")


def plot_internet_type_churn(df: pd.DataFrame) -> None:
    ct = pd.crosstab(df["internet_type"], df["customer_status"], normalize="index") * 100
    ax = ct[["Churned", "Stayed"]].plot(
        kind="bar", figsize=(8, 4), color=["#e74c3c", "#2ecc71"], alpha=0.8
    )
    ax.set_xlabel("Internet Type")
    ax.set_ylabel("% Customers")
    ax.set_title("Customer Status by Internet Type")
    ax.legend(title="Status")
    ax.tick_params(axis="x", rotation=30)
    fig = ax.get_figure()
    fig.tight_layout()
    _save(fig, "internet_type_vs_churn")


# ---------------------------------------------------------------------------
# Main EDA runner
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> None:
    """Run full EDA and save figures to reports/figures/."""
    logger.info("Starting EDA...")
    describe_dataset(df)

    categorical_cols = [
        "gender", "married", "phone_service", "multiple_lines",
        "internet_service", "internet_type", "streaming_tv", "streaming_movies",
        "streaming_music", "unlimited_data", "payment_method",
    ]
    numeric_cols = [
        "age", "tenure_in_months", "number_of_referrals",
        "avg_monthly_long_distance_charges", "avg_monthly_gb_download",
        "total_charges", "total_refunds", "total_extra_data_charges",
        "total_long_distance_charges", "total_revenue",
    ]

    plot_churn_rate_bars(df, categorical_cols)
    plot_numeric_distributions(df, numeric_cols)
    plot_correlation_heatmap(df)
    plot_churn_by_tenure(df)
    plot_revenue_vs_churn(df)
    plot_internet_type_churn(df)

    logger.info("EDA complete. Figures in: %s", FIGURES_DIR)
