import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.loader import load_config, load_modeling_dataset
from src.evaluation.metrics import compute_all_metrics, compute_net_value, find_optimal_threshold
from src.features.engineering import prepare_features
from src.models.trainer import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Telecom Churn Model Training")
    parser.add_argument(
        "--models", nargs="+",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        default=None,
        help="Subset of models to train (default: all)",
    )
    parser.add_argument("--trials", type=int, default=None, help="Override Optuna trial count")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = args.config or (Path(__file__).parents[1] / "config" / "config.yaml")
    config = load_config(config_path)

    if args.trials:
        config["training"]["n_optuna_trials"] = args.trials
        logger.info("Optuna trials overridden to %d", args.trials)

    db_path = Path(__file__).parents[1] / config["data"]["db_path"]

    logger.info("Loading modeling dataset...")
    df = load_modeling_dataset(
        db_path=db_path,
        modeling_population=config["data"]["modeling_population"],
    )

    logger.info("Preparing features...")
    X, y, numeric_cols, categorical_cols = prepare_features(
        df, exclude_cols=config["features"]["exclude_columns"]
    )

    logger.info(
        "Feature matrix: %d rows × %d cols | Churn rate: %.1f%%",
        *X.shape, y.mean() * 100,
    )

    results, champion = train_and_evaluate(
        X=X, y=y,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        config=config,
        model_names=args.models,
    )

    logger.info("=" * 60)
    logger.info("Training summary")
    logger.info("=" * 60)
    for model_name, res in sorted(results.items(), key=lambda x: x[1]["metrics"]["roc_auc"], reverse=True):
        m = res["metrics"]
        logger.info(
            "  %-20s  ROC-AUC=%.4f  F2=%.4f  Recall=%.4f  Precision=%.4f  Threshold=%.3f",
            model_name, m["roc_auc"], m["f2"], m["recall"], m["precision"], m["threshold"],
        )

    logger.info("")
    logger.info("Champion model: %s", champion)
    champ = results[champion]
    logger.info(
        "  MLflow run_id: %s", champ.get("run_id", "N/A")
    )
    logger.info(
        "  Net campaign value (test, 10%% response rate assumption): $%.0f",
        champ["net_value"]["net_value_usd"],
    )

    if champ["feature_importances"] is not None:
        logger.info("\nTop-10 features (%s):", champion)
        fi = champ["feature_importances"].head(10)
        for _, row in fi.iterrows():
            logger.info("  %-40s  %.4f", row["feature"], row["importance"])

    logger.info("\nTo register the champion model, run:")
    logger.info("  uv run python scripts/register_model.py")


if __name__ == "__main__":
    main()
