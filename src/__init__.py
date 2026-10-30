"""
Linear Regression From Scratch Package
======================================
Modular implementation of Linear Regression and evaluation metrics using pure NumPy.
"""

from .linear_regression import LinearRegressionScratch
from .preprocessing import StandardScalerScratch, load_california_housing_data
from .evaluation import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    evaluate_model,
)
from .visualization import (
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_cost_convergence,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_simple_linear_regression,
    plot_model_comparison,
)

__all__ = [
    "LinearRegressionScratch",
    "StandardScalerScratch",
    "load_california_housing_data",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "evaluate_model",
    "plot_feature_distributions",
    "plot_correlation_heatmap",
    "plot_cost_convergence",
    "plot_actual_vs_predicted",
    "plot_residuals",
    "plot_simple_linear_regression",
    "plot_model_comparison",
]
