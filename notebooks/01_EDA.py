"""
EDA Notebook — Telecom Churn (AllLife Communications)
Can be run as a Databricks notebook or standalone Python script.

Databricks: attach to cluster, run with Run All.
Local: python notebooks/01_EDA.py
"""

# COMMAND ----------
# %% [markdown]
# # EDA — AllLife Communications Telecom Churn
#
# **Objective:** Understand the structure, quality, and predictive signals in the
# Q2 2022 customer snapshot before modelling.
#
# **Key analytical guardrails (from README):**
# - `churn_category` and `churn_reason` are post-outcome fields → excluded from predictors.
# - `Joined` customers are excluded from the modelling population.
# - All billing fields are cumulative end-of-quarter snapshots.

# COMMAND ----------
# %% Setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import logging
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.loader import load_config, load_modeling_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
FIGURES_DIR = Path.cwd().parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(__file__).parents[1] / "config" / "config.yaml"
config = load_config(CONFIG_PATH)
DB_PATH = (Path(__file__).parents[1] / config["data"]["db_path"]).resolve()

# COMMAND ----------
# %% Load full dataset (all statuses) for EDA
conn = sqlite3.connect(DB_PATH)
df_all = pd.read_sql_query("""
    SELECT
        c.customer_id, c.gender, c.age, c.married, c.zip_code,
        c.number_of_referrals, c.tenure_in_months,
        sf.phone_service, sf.avg_monthly_long_distance_charges, sf.multiple_lines,
        sf.internet_service, sf.internet_type, sf.avg_monthly_gb_download,
        sf.streaming_tv, sf.streaming_movies, sf.streaming_music, sf.unlimited_data,
        bf.payment_method, bf.total_charges, bf.total_refunds,
        bf.total_extra_data_charges, bf.total_long_distance_charges, bf.total_revenue,
        zp.population AS zip_population,
        co.customer_status, co.churn_category, co.churn_reason
    FROM customers c
    JOIN service_features sf   ON c.customer_id = sf.customer_id
    JOIN billing_features bf   ON c.customer_id = bf.customer_id
    JOIN churn_outcomes co     ON c.customer_id = co.customer_id
    LEFT JOIN zip_population zp ON c.zip_code = zp.zip_code
""", conn)
conn.close()

print(f"Full dataset: {df_all.shape[0]:,} rows × {df_all.shape[1]} cols")
print(df_all["customer_status"].value_counts())

# COMMAND ----------
# %% [markdown]
# ## 1. Dataset overview

# COMMAND ----------
# %% Missing values
null_pct = df_all.isnull().mean().sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]
print("Columns with missing values:")
print(null_pct.to_string() if not null_pct.empty else "None")

# COMMAND ----------
# %% Class distribution (all statuses)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = df_all["customer_status"].value_counts()
axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=["#e74c3c", "#2ecc71", "#3498db"], startangle=90)
axes[0].set_title("Customer Status Distribution (All)")

df_model = df_all[df_all["customer_status"].isin(["Churned", "Stayed"])].copy()
df_model["churn_label"] = (df_model["customer_status"] == "Churned").astype(int)

churn_counts = df_model["customer_status"].value_counts()
axes[1].bar(churn_counts.index, churn_counts.values,
            color=["#e74c3c", "#2ecc71"], alpha=0.85)
for i, (name, val) in enumerate(churn_counts.items()):
    axes[1].text(i, val + 20, f"{val:,}\n({val/len(df_model)*100:.1f}%)",
                ha="center", fontsize=10)
axes[1].set_title("Modelling Population (Churned vs Stayed)")
axes[1].set_ylabel("Customers")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "00_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Churn rate (modelling pop): {df_model['churn_label'].mean()*100:.1f}%")

# COMMAND ----------
# %% [markdown]
# **Finding:** ~26% churn rate — moderate class imbalance. We'll use class_weight='balanced'
# in tree models and tune the decision threshold rather than relying on 0.5.

# COMMAND ----------
# %% Churn by internet type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ct = pd.crosstab(df_model["internet_type"], df_model["customer_status"])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct[["Churned", "Stayed"]].plot(
    kind="bar", ax=axes[0], color=["#e74c3c", "#2ecc71"], alpha=0.85
)
axes[0].set_title("Customer Status by Internet Type")
axes[0].set_ylabel("% Customers")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend(title="Status")

# Churn rate
churn_by_internet = df_model.groupby("internet_type")["churn_label"].mean().sort_values(ascending=False)
axes[1].bar(churn_by_internet.index, churn_by_internet.values * 100,
            color="#e74c3c", alpha=0.8)
axes[1].axhline(df_model["churn_label"].mean() * 100, color="grey", linestyle="--",
                label=f"Overall: {df_model['churn_label'].mean()*100:.1f}%")
axes[1].set_title("Churn Rate by Internet Type")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend()

fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_internet_type_churn.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(churn_by_internet.to_string())

# COMMAND ----------
# %% [markdown]
# **Finding:** Fiber Optic customers churn at a ~40% rate — roughly 2× the overall rate.
# This aligns with competitive pressure in high-speed internet markets.

# COMMAND ----------
# %% Churn by tenure
df_model["tenure_bin"] = pd.cut(
    df_model["tenure_in_months"],
    bins=[0, 6, 12, 24, 36, 60, np.inf],
    labels=["0-6m", "7-12m", "13-24m", "25-36m", "37-60m", "60m+"],
)
tenure_churn = df_model.groupby("tenure_bin", observed=True)["churn_label"].agg(["mean", "count"])
tenure_churn.columns = ["churn_rate", "n"]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(tenure_churn.index.astype(str), tenure_churn["churn_rate"] * 100,
              color="#e74c3c", alpha=0.8)
ax.axhline(df_model["churn_label"].mean() * 100, color="grey", linestyle="--",
           label=f"Overall {df_model['churn_label'].mean()*100:.1f}%")
for bar, (_, row) in zip(bars, tenure_churn.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"n={row['n']:,}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Tenure Band")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Customer Tenure")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "02_tenure_churn.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(tenure_churn.to_string())

# COMMAND ----------
# %% [markdown]
# **Finding:** Churn is heavily concentrated in the first 12 months (~50% churn rate),
# dropping sharply for long-tenured customers. Tenure is likely the strongest predictor.

# COMMAND ----------
# %% Numeric distributions: churned vs stayed
num_cols = [
    "age", "tenure_in_months", "number_of_referrals",
    "avg_monthly_long_distance_charges", "avg_monthly_gb_download",
    "total_charges", "total_revenue",
]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
palette = {"Churned": "#e74c3c", "Stayed": "#2ecc71"}

for i, col in enumerate(num_cols):
    ax = axes[i]
    for status in ["Stayed", "Churned"]:
        vals = df_model[df_model["customer_status"] == status][col].dropna()
        ax.hist(vals, bins=35, alpha=0.55, label=status, color=palette[status], density=True)
    ax.set_title(col.replace("_", " ").title(), fontsize=10)
    ax.legend(fontsize=8)

for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Numeric Feature Distributions: Churned vs Stayed", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "03_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# COMMAND ----------
# %% Churn reasons (EDA only - must NOT be used as features)
reason_counts = df_all[df_all["customer_status"] == "Churned"]["churn_reason"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(reason_counts.index, reason_counts.values, color="#e74c3c", alpha=0.8)
ax.set_xlabel("Count")
ax.set_title("Top 15 Churn Reasons (EDA reference only — NOT a predictor)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "04_churn_reasons.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Top churn reasons:")
print(reason_counts.to_string())

# COMMAND ----------
# %% Categorical churn rates
cat_cols = [
    "gender", "married", "phone_service", "multiple_lines",
    "internet_service", "streaming_tv", "streaming_movies",
    "streaming_music", "unlimited_data", "payment_method",
]
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.flatten()

overall_rate = df_model["churn_label"].mean() * 100

for i, col in enumerate(cat_cols):
    summary = df_model.groupby(col)["churn_label"].agg(["mean", "count"])
    summary.columns = ["churn_rate", "n"]
    summary = summary.sort_values("churn_rate", ascending=True)
    ax = axes[i]
    bars = ax.barh(summary.index.astype(str), summary["churn_rate"] * 100,
                   color="#e74c3c", alpha=0.75)
    ax.axvline(overall_rate, color="grey", linestyle="--", linewidth=1)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"n={row['n']:,}", va="center", fontsize=7)
    ax.set_title(col.replace("_", " ").title(), fontsize=9)
    ax.set_xlabel("Churn Rate (%)", fontsize=8)

for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Churn Rate by Categorical Feature (grey line = overall rate)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "05_categorical_churn_rates.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# COMMAND ----------
# %% Correlation with churn label
df_enc = df_model.copy()
bool_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
for col in df_enc.select_dtypes("object"):
    df_enc[col] = df_enc[col].map(bool_map).fillna(df_enc[col])
df_enc = pd.get_dummies(df_enc.select_dtypes(include=["number", "object"]), drop_first=True)

corr = df_enc.corrwith(df_enc["churn_label"]).drop("churn_label").sort_values()
top_bottom = pd.concat([corr.head(10), corr.tail(10)])

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_bottom.values]
ax.barh(top_bottom.index, top_bottom.values, color=colors, alpha=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Churn Label")
ax.set_title("Top/Bottom Features by Correlation with Churn")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "06_feature_correlation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Top correlations with churn:")
print(corr.tail(10).to_string())
print("\nTop negative correlations (churn protective):")
print(corr.head(10).to_string())

# COMMAND ----------
# %% Summary statistics for modeling population
print("\n=== MODELING POPULATION SUMMARY ===")
print(f"Total customers: {len(df_model):,}")
print(f"Churned: {df_model['churn_label'].sum():,} ({df_model['churn_label'].mean()*100:.1f}%)")
print(f"Stayed: {(df_model['churn_label']==0).sum():,} ({(1-df_model['churn_label'].mean())*100:.1f}%)")
print(f"Joined (excluded): {(df_all['customer_status']=='Joined').sum():,}")
print("\nNumeric summary:")
print(df_model[[
    "age", "tenure_in_months", "number_of_referrals",
    "total_charges", "total_revenue", "avg_monthly_gb_download"
]].describe().round(2).to_string())

print("\nAll EDA figures saved to:", FIGURES_DIR)
