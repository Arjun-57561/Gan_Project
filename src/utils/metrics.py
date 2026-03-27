"""Metrics computation utilities."""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc_roc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except:
            metrics["auc_roc"] = 0.0
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
) -> str:
    """Get detailed classification report."""
    return classification_report(y_true, y_pred, target_names=target_names)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """Compute per-class metrics."""
    metrics = {}
    for class_id in range(num_classes):
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        
        metrics[class_id] = {
            "precision": (y_pred_binary & y_true_binary).sum() / max(y_pred_binary.sum(), 1),
            "recall": (y_pred_binary & y_true_binary).sum() / max(y_true_binary.sum(), 1),
            "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }
    
    return metrics
