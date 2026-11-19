"""
SHAP-based model explainability.

Global importance (summary plot) answers "which features drive churn overall?"
Local explanation (waterfall/force plot) answers "why is THIS customer flagged?"

XGBoost / LightGBM / RandomForest → TreeExplainer (exact, fast).
LogisticRegression → LinearExplainer (exact for linear models).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_explainer(clf, X_background: np.ndarray):
    """Return the appropriate SHAP explainer for the classifier."""
    import shap

    model_type = type(clf).__name__
    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(clf)
    elif model_type == "LogisticRegression":
        return shap.LinearExplainer(clf, X_background)
    else:
        logger.warning("No specific SHAP explainer for %s — falling back to KernelExplainer (slow).", model_type)
        return shap.KernelExplainer(clf.predict_proba, shap.sample(X_background, 100))


def compute_shap_values(
    pipeline,
    X: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    use_ohe: bool = False,
    max_display: int = 20,
    figures_dir: Optional[Path] = None,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Compute SHAP values for the champion model and save summary plots.

    Returns a DataFrame with mean |SHAP| per feature (global importance ranking).
    """
    import shap

    if figures_dir is None:
        figures_dir = Path("reports/figures")
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    X_transformed = pre.transform(X)

    if use_ohe:
        ohe = pre.named_transformers_["cat"].named_steps["encoder"]
        cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
    else:
        cat_names = categorical_cols

    feature_names = numeric_cols + cat_names

    if X_transformed.shape[1] != len(feature_names):
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]
        logger.warning("Feature name count mismatch — using generic names.")

    explainer = _get_explainer(clf, X_transformed)

    logger.info("Computing SHAP values for %d samples...", len(X_transformed))
    shap_values = explainer.shap_values(X_transformed)

    # Tree models with multi-output return a list; take the positive-class slice
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # --- Global summary bar plot ---
    fig, ax = plt.subplots(figsize=(9, max(5, min(max_display, len(feature_names)) * 0.4)))
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(max_display)
    mean_abs[::-1].plot(kind="barh", ax=ax, color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = figures_dir / f"13_shap_importance_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP bar plot saved → %s", out_path)

    # --- Beeswarm summary plot (top features) ---
    try:
        top_features = mean_abs.index.tolist()
        shap_top = shap_df[top_features].values
        X_top = pd.DataFrame(X_transformed, columns=feature_names)[top_features]

        fig_bee, ax_bee = plt.subplots(figsize=(10, max(5, len(top_features) * 0.4)))
        shap.summary_plot(
            shap_top,
            X_top,
            feature_names=top_features,
            show=False,
            max_display=max_display,
            plot_size=None,
        )
        bee_path = figures_dir / f"14_shap_beeswarm_{model_name}.png"
        plt.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP beeswarm plot saved → %s", bee_path)
    except Exception as e:
        logger.warning("Could not generate beeswarm plot: %s", e)

    importance_df = (
        shap_df.abs().mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return importance_df
