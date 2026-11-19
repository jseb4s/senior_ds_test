"""
Evaluation utilities.

Business context:
- False Negatives (missed churners) → lost revenue from preventable churn.
- False Positives (wrongly targeted stayers) → wasted campaign spend ($4/contact).
- At $180 margin per retained customer and $4 cost per contact, the break-even
  precision at a given recall is determined by the ROI equation:
      precision_min = cost / (cost + margin) ≈ 2.2%
  This is very low, meaning ROC-AUC and F1 are still the primary drivers, but
  the model must be calibrated for threshold selection.
- Primary metric: ROC-AUC (threshold-invariant, handles class imbalance well).
- Secondary: F2 score (weights recall 2× over precision, reflecting business cost
  asymmetry where missing a churner is more expensive than a wasted contact).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    beta: float = 2.0,
) -> dict:
    """Compute a full suite of classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
    }
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f2",
    beta: float = 2.0,
) -> float:
    """
    Find the probability threshold that maximises the chosen metric.

    strategy options:
    - 'f2': maximises F-beta (beta=2 weights recall higher — good for churn)
    - 'youden': Youden's J (balances sensitivity + specificity)
    - 'precision_recall': maximises F1 on precision-recall curve
    """
    thresholds = np.linspace(0.05, 0.95, 181)

    if strategy == "f2":
        scores = [
            fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
            for t in thresholds
        ]
    elif strategy == "youden":
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        return float(roc_thresholds[np.argmax(j_scores)])
    else:
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0,
        )
        return float(pr_thresholds[np.argmax(f1_scores[:-1])])

    best_t = float(thresholds[np.argmax(scores)])
    logger.info("Optimal threshold (%s): %.3f (score=%.4f)", strategy, best_t, max(scores))
    return best_t


def compute_net_value(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    contact_cost: float = 4.0,
    retention_margin: float = 180.0,
    response_rate: float = 0.10,
) -> dict:
    """
    Estimate expected campaign net value under simplifying assumptions.

    response_rate: assumed fraction of correctly targeted churners who are retained
    by the campaign. This is NOT derivable from the churn dataset alone — it requires
    a randomised experiment or observational causal study. 10% is a conservative
    placeholder that should be validated.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_contacts = tp + fp
    expected_retentions = tp * response_rate
    gross_value = expected_retentions * retention_margin
    total_cost = total_contacts * contact_cost
    net_value = gross_value - total_cost

    return {
        "total_contacts": int(total_contacts),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "expected_retentions": expected_retentions,
        "gross_value_usd": gross_value,
        "total_cost_usd": total_cost,
        "net_value_usd": net_value,
        "roi_pct": (net_value / total_cost * 100) if total_cost > 0 else 0.0,
        "assumed_response_rate": response_rate,
    }


def log_metrics_summary(metrics: dict, prefix: str = "") -> None:
    p = f"{prefix}_" if prefix else ""
    logger.info(
        "%sROC-AUC=%.4f | AvgPrecision=%.4f | F1=%.4f | F2=%.4f | "
        "Precision=%.4f | Recall=%.4f | Threshold=%.3f",
        p,
        metrics["roc_auc"],
        metrics["avg_precision"],
        metrics["f1"],
        metrics["f2"],
        metrics["precision"],
        metrics["recall"],
        metrics["threshold"],
    )
