import argparse
import logging
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Register champion model in MLflow Model Registry")
    p.add_argument("--run-id", type=str, default=None,
                   help="MLflow run_id to register (default: best run by ROC-AUC)")
    p.add_argument("--config", type=str, default=None,
                   help="Path to config.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = args.config or (Path(__file__).parents[1] / "config" / "config.yaml")
    config = load_config(config_path)

    tracking_uri = config["mlflow"]["tracking_uri"]
    exp_name = config["mlflow"]["experiment_name"]
    registry_name = config["mlflow"]["model_registry_name"]

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{exp_name}' not found. Run train.py first.")

    if args.run_id:
        run = client.get_run(args.run_id)
        champion_run_id = args.run_id
        champion_algo = run.data.tags.get("model_name", "unknown")
        roc_auc = run.data.metrics.get("roc_auc", float("nan"))
        logger.info("Using explicit run_id: %s (%s, ROC-AUC=%.4f)", champion_run_id, champion_algo, roc_auc)
    else:
        runs = client.search_runs(
            exp.experiment_id,
            order_by=["metrics.roc_auc DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No runs found. Run train.py first.")
        best_run = runs[0]
        champion_run_id = best_run.info.run_id
        champion_algo = best_run.data.tags.get("model_name", "unknown")
        roc_auc = best_run.data.metrics.get("roc_auc", float("nan"))
        logger.info(
            "Best run by ROC-AUC: %s (%s, ROC-AUC=%.4f)",
            champion_run_id, champion_algo, roc_auc,
        )

    logged_models = client.search_logged_models(experiment_ids=[exp.experiment_id])
    model_uri = None
    for m in logged_models:
        if m.source_run_id == champion_run_id:
            model_uri = m.model_uri
            break

    if model_uri is None:
        raise RuntimeError(
            f"No logged model found for run_id={champion_run_id}. "
            "Re-run train.py to ensure the model was saved correctly."
        )

    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
    logger.info(
        "Model registered: name=%s | version=%s | run_id=%s",
        registry_name, mv.version, champion_run_id,
    )

    client.set_registered_model_alias(
        name=registry_name,
        alias="champion",
        version=mv.version,
    )
    logger.info("Alias 'champion' → %s v%s", registry_name, mv.version)

    client.set_model_version_tag(
        name=registry_name,
        version=mv.version,
        key="algorithm",
        value=champion_algo,
    )
    logger.info("Tag algorithm=%s added to v%s", champion_algo, mv.version)

    logger.info("\nDone. Load champion in production with:")
    logger.info("  mlflow.sklearn.load_model('models:/%s@champion')", registry_name)


if __name__ == "__main__":
    main()
