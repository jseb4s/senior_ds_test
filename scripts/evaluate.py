"""
Post-training evaluation: ROC curves, PR curves, feature importances, threshold analysis.
Loads the best model from MLflow and generates report figures.
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.loader import load_config, load_modeling_dataset
from src.evaluation.metrics import compute_all_metrics, compute_net_value
from src.evaluation.shap_analysis import compute_shap_values
from src.features.engineering import prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parents[1] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "logistic_regression": "#3498db",
    "random_forest": "#2ecc71",
    "xgboost": "#e74c3c",
    "lightgbm": "#f39c12",
}


def load_all_runs(tracking_uri: str, experiment_name: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(exp.experiment_id)
    return pd.DataFrame([
        {
            "run_id": r.info.run_id,
            "model_name": r.data.tags.get("model_name", ""),
            **{k: v for k, v in r.data.metrics.items()},
        }
        for r in runs
    ])


def plot_roc_comparison(models_probs: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, y_prob in models_probs.items():
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})",
                color=COLORS.get(model_name, "grey"), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — All Models (Test Set)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved.")


def plot_pr_comparison(models_probs: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()
    for model_name, y_prob in models_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec, prec, label=model_name, color=COLORS.get(model_name, "grey"), linewidth=2)
    ax.axhline(baseline, color="grey", linestyle="--", label=f"Baseline (={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves — All Models (Test Set)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved.")


def plot_feature_importance(fi_df: pd.DataFrame, model_name: str, top_n: int = 20) -> None:
    fi = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"09_feature_importance_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance saved for %s.", model_name)


def plot_threshold_analysis(y_test: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    thresholds = np.linspace(0.05, 0.95, 181)
    precisions, recalls, f1s, f2s = [], [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f2 = (1 + 4) * p * r / (4 * p + r) if (4 * p + r) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        f2s.append(f2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#3498db")
    ax.plot(thresholds, recalls, label="Recall", color="#e74c3c")
    ax.plot(thresholds, f1s, label="F1", color="#2ecc71", linestyle="--")
    ax.plot(thresholds, f2s, label="F2 (β=2)", color="#f39c12", linestyle="--")

    best_f2_idx = np.argmax(f2s)
    ax.axvline(thresholds[best_f2_idx], color="grey", linestyle=":", linewidth=1.5,
               label=f"Best F2 threshold={thresholds[best_f2_idx]:.3f}")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"10_threshold_analysis_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Threshold analysis saved for %s.", model_name)


def plot_calibration(models_probs: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for model_name, y_prob in models_probs.items():
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, "s-", label=model_name, color=COLORS.get(model_name, "grey"))
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — All Models")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration curves saved.")


def plot_score_distribution(models_probs: dict, y_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, len(models_probs), figsize=(5 * len(models_probs), 4))
    if len(models_probs) == 1:
        axes = [axes]

    for ax, (model_name, y_prob) in zip(axes, models_probs.items()):
        for label, color, name in [(0, "#2ecc71", "Stayed"), (1, "#e74c3c", "Churned")]:
            ax.hist(y_prob[y_test == label], bins=30, alpha=0.6, density=True,
                    color=color, label=name)
        ax.set_title(model_name)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Score Distribution: Churned vs Stayed", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Score distributions saved.")


def main():
    config_path = Path(__file__).parents[1] / "config" / "config.yaml"
    config = load_config(config_path)

    db_path = (Path(__file__).parents[1] / config["data"]["db_path"]).resolve()
    df = load_modeling_dataset(db_path, config["data"]["modeling_population"])
    X, y, numeric_cols, categorical_cols = prepare_features(df, config["features"]["exclude_columns"])
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        stratify=y,
        random_state=config["training"]["random_state"],
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    runs = client.search_runs(exp.experiment_id, order_by=["metrics.roc_auc DESC"])

    models_probs = {}
    fi_by_model = {}

    for run in runs:
        model_name = run.data.tags.get("model_name", "")
        if not model_name:
            continue
        logger.info("Loading model: %s (run_id=%s)", model_name, run.info.run_id)
        try:
            pipeline = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            models_probs[model_name] = y_prob
        except Exception as e:
            logger.warning("Could not load %s: %s", model_name, e)

    if not models_probs:
        logger.error("No models loaded. Run train.py first.")
        return

    plot_roc_comparison(models_probs, y_test.values)
    plot_pr_comparison(models_probs, y_test.values)
    plot_calibration(models_probs, y_test.values)
    plot_score_distribution(models_probs, y_test.values)

    # Champion model details
    champion = max(models_probs, key=lambda m: roc_auc_score(y_test, models_probs[m]))
    logger.info("Generating detailed plots for champion: %s", champion)

    champ_prob = models_probs[champion]
    plot_threshold_analysis(y_test.values, champ_prob, champion)

    # Feature importances from artifact
    for run in runs:
        m = run.data.tags.get("model_name", "")
        if m != champion:
            continue
        try:
            art_path = mlflow.artifacts.download_artifacts(
                f"runs:/{run.info.run_id}/{champion}_feature_importance.csv"
            )
            fi_df = pd.read_csv(art_path)
            plot_feature_importance(fi_df, champion)
        except Exception as e:
            logger.warning("Could not load feature importances: %s", e)

    # Print final metrics table
    logger.info("\n%s", "=" * 80)
    logger.info("FINAL MODEL COMPARISON (Test Set)")
    logger.info("%-22s  %-10s  %-10s  %-10s  %-10s  %-10s", "Model", "ROC-AUC", "F2", "Recall", "Precision", "Threshold")
    logger.info("-" * 80)
    for model_name, y_prob in sorted(
        models_probs.items(), key=lambda x: roc_auc_score(y_test, x[1]), reverse=True
    ):
        metrics = compute_all_metrics(y_test.values, y_prob)
        logger.info(
            "%-22s  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.3f",
            model_name, metrics["roc_auc"], metrics["f2"],
            metrics["recall"], metrics["precision"], metrics["threshold"],
        )

    # --- SHAP explainability for champion ---
    logger.info("Computing SHAP values for champion model: %s", champion)
    champ_pipeline = None
    champ_use_ohe = champion == "logistic_regression"
    for run in runs:
        if run.data.tags.get("model_name", "") == champion:
            try:
                champ_pipeline = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
            except Exception as e:
                logger.warning("Could not reload champion pipeline for SHAP: %s", e)
            break

    if champ_pipeline is not None:
        try:
            shap_importance = compute_shap_values(
                pipeline=champ_pipeline,
                X=X_test,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                use_ohe=champ_use_ohe,
                figures_dir=FIGURES_DIR,
                model_name=champion,
            )
            logger.info("Top-10 features by mean |SHAP|:")
            for _, row in shap_importance.head(10).iterrows():
                logger.info("  %-40s  %.4f", row["feature"], row["mean_abs_shap"])
        except Exception as e:
            logger.warning("SHAP analysis failed: %s", e)

    # --- Logistic Regression odds ratios (interpretable communication model) ---
    for run in runs:
        if run.data.tags.get("model_name", "") != "logistic_regression":
            continue
        try:
            lr_pipeline = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
            _plot_lr_odds_ratios(lr_pipeline, numeric_cols, categorical_cols)
        except Exception as e:
            logger.warning("Could not extract LR odds ratios: %s", e)
        break

    logger.info("\nAll figures saved to: %s", FIGURES_DIR)


def _plot_lr_odds_ratios(pipeline, numeric_cols: list, categorical_cols: list, top_n: int = 20) -> None:
    """
    Extract logistic regression coefficients as odds ratios and save a plot.

    Odds ratio > 1 → feature increases churn probability.
    Odds ratio < 1 → feature is protective.
    Coefficients are from the standardised feature space; direction is reliable but
    magnitude comparisons across features with different scales should be treated with care.
    """
    import numpy as np

    clf = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]

    ohe = pre.named_transformers_["cat"].named_steps["encoder"]
    cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
    feature_names = numeric_cols + cat_names

    coefs = clf.coef_[0]
    if len(coefs) != len(feature_names):
        feature_names = [f"f{i}" for i in range(len(coefs))]

    odds_ratios = np.exp(coefs)
    df_or = (
        pd.DataFrame({"feature": feature_names, "odds_ratio": odds_ratios})
        .assign(log_or=lambda d: np.log(d["odds_ratio"]))
        .reindex(pd.Series(np.abs(np.log(odds_ratios))).sort_values(ascending=False).index)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Save CSV for stakeholder reporting
    or_csv = FIGURES_DIR.parent / "logistic_regression_odds_ratios.csv"
    df_or.to_csv(or_csv, index=False)
    logger.info("Odds ratios saved → %s", or_csv)

    # Plot
    colors = ["#e74c3c" if v > 1 else "#3498db" for v in df_or["odds_ratio"]]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    ax.barh(df_or["feature"][::-1], df_or["odds_ratio"][::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Odds Ratio  (>1 = increases churn risk, <1 = protective)")
    ax.set_title(f"Logistic Regression — Top {top_n} Features by |log OR|")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = FIGURES_DIR / "15_logistic_regression_odds_ratios.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Odds ratio plot saved → %s", out_path)

    logger.info("Top-10 LR odds ratios:")
    for _, row in df_or.head(10).iterrows():
        direction = "↑ churn" if row["odds_ratio"] > 1 else "↓ churn"
        logger.info("  %-40s  OR=%.3f  %s", row["feature"], row["odds_ratio"], direction)


if __name__ == "__main__":
    main()
