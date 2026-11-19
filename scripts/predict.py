#!/usr/bin/env python
"""
Batch scoring script.

Loads the champion model from MLflow and scores a customer dataset,
producing churn probabilities and binary predictions.

Usage:
    # Score the held-out test set (default — useful for validation)
    python scripts/predict.py

    # Score a CSV file of new customers
    python scripts/predict.py --input path/to/customers.csv --output scores.csv

    # Use a specific MLflow run instead of the best run by ROC-AUC
    python scripts/predict.py --run-id <mlflow_run_id>

Output columns appended to the input data:
    churn_probability  — model score in [0, 1]
    churn_prediction   — 1 if probability >= threshold, else 0
    risk_tier          — High / Medium / Low based on probability quantiles

Databricks usage:
    - Set MLFLOW_TRACKING_URI to the Databricks tracking server.
    - Replace db_path with the DBFS or Unity Catalog table path.
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.loader import load_config, load_modeling_dataset
from src.features.engineering import prepare_features
from src.evaluation.metrics import find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_champion_pipeline(tracking_uri: str, experiment_name: str, run_id: str = None):
    """Load the best-performing pipeline from MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    if run_id:
        logger.info("Loading model from explicit run_id: %s", run_id)
        pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        run = client.get_run(run_id)
        threshold = run.data.metrics.get("threshold", 0.5)
        model_name = run.data.tags.get("model_name", "unknown")
        return pipeline, float(threshold), model_name

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' not found. Run train.py first."
        )

    runs = client.search_runs(
        exp.experiment_id,
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Run train.py first.")

    best_run = runs[0]
    model_name = best_run.data.tags.get("model_name", "unknown")
    threshold = best_run.data.metrics.get("threshold", 0.5)
    logger.info(
        "Champion model: %s | run_id=%s | ROC-AUC=%.4f | threshold=%.3f",
        model_name,
        best_run.info.run_id,
        best_run.data.metrics.get("roc_auc", float("nan")),
        threshold,
    )

    pipeline = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
    return pipeline, float(threshold), model_name


def _assign_risk_tier(probs: np.ndarray) -> pd.Series:
    """Assign High / Medium / Low risk labels using fixed thresholds."""
    tiers = pd.Series("Low", index=range(len(probs)))
    tiers[probs >= 0.6] = "High"
    tiers[(probs >= 0.3) & (probs < 0.6)] = "Medium"
    return tiers


def score_dataframe(
    df: pd.DataFrame,
    pipeline,
    threshold: float,
    exclude_cols: list,
    customer_id_col: str = "customer_id",
) -> pd.DataFrame:
    """Score a raw DataFrame and return predictions merged with customer IDs."""
    X, _, _, _ = prepare_features(df, exclude_cols=exclude_cols)
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    scores = df[[customer_id_col]].copy() if customer_id_col in df.columns else pd.DataFrame()
    scores["churn_probability"] = np.round(y_prob, 4)
    scores["churn_prediction"] = y_pred
    scores["risk_tier"] = _assign_risk_tier(y_prob).values

    churn_rate = y_pred.mean() * 100
    logger.info(
        "Scored %d customers | Predicted churn rate: %.1f%% | Threshold: %.3f",
        len(df), churn_rate, threshold,
    )
    high = (scores["risk_tier"] == "High").sum()
    med = (scores["risk_tier"] == "Medium").sum()
    low = (scores["risk_tier"] == "Low").sum()
    logger.info("Risk tiers — High: %d | Medium: %d | Low: %d", high, med, low)

    return scores


def parse_args():
    p = argparse.ArgumentParser(description="Telecom Churn Batch Scoring")
    p.add_argument("--input", type=str, default=None,
                   help="Path to input CSV (default: score the SQLite modeling population)")
    p.add_argument("--output", type=str, default=None,
                   help="Path to write scored CSV (default: reports/predictions.csv)")
    p.add_argument("--run-id", type=str, default=None,
                   help="MLflow run_id to load (default: best run by ROC-AUC)")
    p.add_argument("--config", type=str, default=None,
                   help="Path to config.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = args.config or (Path(__file__).parents[1] / "config" / "config.yaml")
    config = load_config(config_path)

    pipeline, threshold, model_name = _load_champion_pipeline(
        tracking_uri=config["mlflow"]["tracking_uri"],
        experiment_name=config["mlflow"]["experiment_name"],
        run_id=args.run_id,
    )

    if args.input:
        logger.info("Loading input from: %s", args.input)
        df = pd.read_csv(args.input)
    else:
        db_path = (Path(__file__).parents[1] / config["data"]["db_path"]).resolve()
        logger.info("Loading from SQLite: %s", db_path)
        df = load_modeling_dataset(
            db_path=db_path,
            modeling_population=config["data"]["modeling_population"],
        )

    scores = score_dataframe(
        df=df,
        pipeline=pipeline,
        threshold=threshold,
        exclude_cols=config["features"]["exclude_columns"],
    )

    out_path = args.output or (Path(__file__).parents[1] / "reports" / "predictions.csv")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_path, index=False)
    logger.info("Predictions written → %s (%d rows)", out_path, len(scores))

    logger.info("\nSample predictions (top 10 by churn probability):")
    logger.info("\n%s", scores.sort_values("churn_probability", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
