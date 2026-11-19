"""
Model training orchestrator with Optuna hyperparameter optimisation + MLflow tracking.

Architecture decisions:
1. Each candidate model gets its own Optuna study logged under a separate MLflow run.
2. Optuna's TPE sampler (default) outperforms random search and grid search on the
   same budget — we use it for all models.
3. Cross-validation inside each Optuna trial to prevent optimistic bias from a single
   train/val split.
4. The final model is retrained on the full training set with best params.
5. MLflow logs: params, metrics, feature importances, the trained pipeline artifact.
6. Model selection: best model by ROC-AUC on the held-out test set.
"""

import logging
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_net_value,
    find_optimal_threshold,
    log_metrics_summary,
)
from src.features.engineering import build_preprocessor

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Optuna objective factories
# ---------------------------------------------------------------------------

def _objective_lr(trial, X_train, y_train, preprocessor, cv, scoring):
    C = trial.suggest_float("C", 1e-4, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    max_iter = trial.suggest_int("max_iter", 200, 1000)

    model = LogisticRegression(
        C=C, solver=solver, max_iter=max_iter,
        class_weight="balanced", random_state=42,
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


def _objective_rf(trial, X_train, y_train, preprocessor, cv, scoring):
    n_estimators = trial.suggest_int("n_estimators", 100, 600)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


def _objective_xgb(trial, X_train, y_train, preprocessor, cv, scoring, scale_pos_weight):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


def _objective_lgbm(trial, X_train, y_train, preprocessor, cv, scoring, scale_pos_weight):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }
    model = lgb.LGBMClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


# ---------------------------------------------------------------------------
# Model builders with best params
# ---------------------------------------------------------------------------

def _build_lr(params, use_ohe: bool = True):
    return LogisticRegression(
        C=params["C"],
        solver=params.get("solver", "lbfgs"),
        max_iter=params.get("max_iter", 500),
        class_weight="balanced", random_state=42,
    ), use_ohe


def _build_rf(params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced", random_state=42, n_jobs=-1,
    ), False


def _build_xgb(params, scale_pos_weight):
    p = {k: v for k, v in params.items()}
    return xgb.XGBClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        min_child_weight=p["min_child_weight"],
        reg_alpha=p["reg_alpha"],
        reg_lambda=p["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
    ), False


def _build_lgbm(params, scale_pos_weight):
    return lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_samples=params["min_child_samples"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, verbose=-1,
    ), False


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list,
    categorical_cols: list,
    config: dict,
    model_names: Optional[list] = None,
) -> dict:
    """
    Runs Optuna-tuned training for each candidate model, tracks with MLflow.
    Returns dict of {model_name: {'pipeline': ..., 'metrics': ..., 'threshold': ...}}.
    """
    cfg_train = config["training"]
    cfg_mlflow = config["mlflow"]

    test_size = cfg_train["test_size"]
    n_trials = cfg_train["n_optuna_trials"]
    timeout = cfg_train["optuna_timeout_seconds"]
    scoring = cfg_train["scoring_metric"]
    cv_folds = cfg_train["cv_folds"]
    random_state = cfg_train["random_state"]

    if model_names is None:
        model_names = config["models"]["candidates"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(
        "Train/test split: %d train, %d test (stratified, test_size=%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info("scale_pos_weight (class imbalance ratio): %.3f", scale_pos_weight)

    mlflow.set_tracking_uri(cfg_mlflow["tracking_uri"])
    mlflow.set_experiment(cfg_mlflow["experiment_name"])

    results = {}

    for model_name in model_names:
        logger.info("=" * 60)
        logger.info("Training model: %s", model_name)

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("model_name", model_name)
            mlflow.log_params({
                "n_optuna_trials": n_trials,
                "cv_folds": cv_folds,
                "scoring_metric": scoring,
                "test_size": test_size,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "scale_pos_weight": scale_pos_weight,
            })

            # Determine whether to use OHE (logistic regression needs it)
            use_ohe = (model_name == "logistic_regression")
            preprocessor = build_preprocessor(numeric_cols, categorical_cols, use_ohe=use_ohe)

            # --- Optuna study ---
            study = optuna.create_study(direction="maximize", study_name=model_name)

            if model_name == "logistic_regression":
                objective = lambda t: _objective_lr(t, X_train, y_train, preprocessor, cv, scoring)
            elif model_name == "random_forest":
                objective = lambda t: _objective_rf(t, X_train, y_train, preprocessor, cv, scoring)
            elif model_name == "xgboost":
                objective = lambda t: _objective_xgb(t, X_train, y_train, preprocessor, cv, scoring, scale_pos_weight)
            elif model_name == "lightgbm":
                objective = lambda t: _objective_lgbm(t, X_train, y_train, preprocessor, cv, scoring, scale_pos_weight)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

            best_params = study.best_params
            best_cv_score = study.best_value
            logger.info(
                "%s best CV %s=%.4f | params=%s",
                model_name, scoring, best_cv_score, best_params,
            )
            mlflow.log_metric(f"cv_{scoring}", best_cv_score)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

            # --- Retrain on full training set with best params ---
            if model_name == "logistic_regression":
                clf, _ = _build_lr(best_params, use_ohe=use_ohe)
            elif model_name == "random_forest":
                clf, _ = _build_rf(best_params)
            elif model_name == "xgboost":
                clf, _ = _build_xgb(best_params, scale_pos_weight)
            elif model_name == "lightgbm":
                clf, _ = _build_lgbm(best_params, scale_pos_weight)

            pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])
            pipeline.fit(X_train, y_train)

            # --- Evaluation ---
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            optimal_threshold = find_optimal_threshold(y_test, y_prob, strategy="f2")
            metrics = compute_all_metrics(y_test, y_prob, threshold=optimal_threshold)
            net_val = compute_net_value(y_test.values, y_prob, optimal_threshold)

            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_metrics({
                "net_value_usd": net_val["net_value_usd"],
                "expected_retentions": net_val["expected_retentions"],
                "roi_pct": net_val["roi_pct"],
            })

            log_metrics_summary(metrics, prefix=model_name)

            # --- Feature importances (tree models) ---
            fi_df = _extract_feature_importances(pipeline, numeric_cols, categorical_cols, use_ohe)
            if fi_df is not None:
                fi_path = f"/tmp/{model_name}_feature_importance.csv"
                fi_df.to_csv(fi_path, index=False)
                mlflow.log_artifact(fi_path)

            # --- Log model ---
            # Use pickle serialization to avoid skops trusted-type friction in dev.
            # In Databricks production, consider mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE.
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )

            results[model_name] = {
                "pipeline": pipeline,
                "metrics": metrics,
                "threshold": optimal_threshold,
                "cv_score": best_cv_score,
                "best_params": best_params,
                "net_value": net_val,
                "feature_importances": fi_df,
                "run_id": mlflow.active_run().info.run_id,
            }

    # --- Select champion model ---
    champion = max(results, key=lambda m: results[m]["metrics"]["roc_auc"])
    logger.info("Champion model: %s (ROC-AUC=%.4f)", champion, results[champion]["metrics"]["roc_auc"])

    return results, champion


def _extract_feature_importances(
    pipeline: Pipeline,
    numeric_cols: list,
    categorical_cols: list,
    use_ohe: bool,
) -> Optional[pd.DataFrame]:
    clf = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]

    try:
        if use_ohe:
            ohe = pre.named_transformers_["cat"].named_steps["encoder"]
            cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
        else:
            cat_names = categorical_cols

        feature_names = numeric_cols + cat_names

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return None

        fi = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return fi
    except Exception as e:
        logger.warning("Could not extract feature importances: %s", e)
        return None
