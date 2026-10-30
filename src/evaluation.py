"""
Model Evaluation Metrics From Scratch Using NumPy
=================================================
This module implements standard regression performance metrics completely
from scratch using NumPy, without relying on sklearn.metrics.

Implemented Metrics:
1. Mean Squared Error (MSE)
2. Root Mean Squared Error (RMSE)
3. Mean Absolute Error (MAE)
4. Coefficient of Determination (R² Score)
"""

from typing import Dict, Union
import numpy as np


def _check_arrays(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and align true and predicted target arrays."""
    y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_true_arr) == 0:
        raise ValueError("Cannot calculate metrics on empty target array.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Shape mismatch: y_true has {len(y_true_arr)} values but y_pred has {len(y_pred_arr)} values."
        )

    return y_true_arr, y_pred_arr


def mean_squared_error(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    """
    Compute Mean Squared Error (MSE):
        MSE = (1 / m) * sum((y_true - y_pred) ** 2)

    Parameters
    ----------
    y_true : array-like of shape (m,)
        Ground truth target values.
    y_pred : array-like of shape (m,)
        Estimated target values.

    Returns
    -------
    mse : float
        Mean squared error loss.
    """
    y_t, y_p = _check_arrays(y_true, y_pred)
    return float(np.mean((y_t - y_p) ** 2))


def root_mean_squared_error(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    """
    Compute Root Mean Squared Error (RMSE):
        RMSE = sqrt(MSE)

    Parameters
    ----------
    y_true : array-like of shape (m,)
        Ground truth target values.
    y_pred : array-like of shape (m,)
        Estimated target values.

    Returns
    -------
    rmse : float
        Root mean squared error.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    """
    Compute Mean Absolute Error (MAE):
        MAE = (1 / m) * sum(|y_true - y_pred|)

    Parameters
    ----------
    y_true : array-like of shape (m,)
        Ground truth target values.
    y_pred : array-like of shape (m,)
        Estimated target values.

    Returns
    -------
    mae : float
        Mean absolute error.
    """
    y_t, y_p = _check_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def r2_score(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> float:
    """
    Compute Coefficient of Determination (R² Score):
        R² = 1 - (SS_res / SS_tot)
        where SS_res = sum((y_true - y_pred) ** 2)
              SS_tot = sum((y_true - mean(y_true)) ** 2)

    Parameters
    ----------
    y_true : array-like of shape (m,)
        Ground truth target values.
    y_pred : array-like of shape (m,)
        Estimated target values.

    Returns
    -------
    r2 : float
        R² score (1.0 indicates perfect prediction, <= 0.0 indicates worse than mean baseline).
    """
    y_t, y_p = _check_arrays(y_true, y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1.0 - (ss_res / ss_tot))


def evaluate_model(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> Dict[str, float]:
    """
    Calculate and return all standard regression metrics in a dictionary.

    Parameters
    ----------
    y_true : array-like of shape (m,)
        Ground truth target values.
    y_pred : array-like of shape (m,)
        Estimated target values.

    Returns
    -------
    metrics : dict
        Dictionary containing 'MSE', 'RMSE', 'MAE', and 'R2'.
    """
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics_table(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> None:
    """
    Print a cleanly formatted terminal table of performance metrics.
    """
    print(f"\n{'=' * 40}")
    print(f"{title:^40}")
    print(f"{'=' * 40}")
    print(f"{'Metric':<20} | {'Value':>15}")
    print(f"{'-' * 20}-+-{'-' * 15}")
    for k, v in metrics.items():
        print(f"{k:<20} | {v:>15.6f}")
    print(f"{'=' * 40}\n")
