"""
Linear Regression From Scratch Using NumPy - Main Execution Pipeline
====================================================================
This script executes the complete end-to-end Machine Learning pipeline:
1. Dataset loading & Exploratory Data Analysis (EDA)
2. Train-test splitting & Leakage-free Standardization
3. Simple Linear Regression from scratch
4. Multiple Linear Regression from scratch with Batch Gradient Descent
5. Evaluation using custom NumPy metrics (MSE, RMSE, MAE, R²)
6. Scikit-learn benchmark comparison and mathematical analysis
7. Visualization generation and export to results/

Author: Senior Machine Learning Engineer
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import custom scratch components
from src.linear_regression import LinearRegressionScratch
from src.preprocessing import StandardScalerScratch, load_california_housing_data
from src.evaluation import evaluate_model, print_metrics_table
from src.visualization import (
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_cost_convergence,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_simple_linear_regression,
    plot_model_comparison,
)


def ensure_directories():
    """Ensure data/ and results/ directories exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def run_pipeline():
    print("=" * 80)
    print("  LINEAR REGRESSION FROM SCRATCH USING NUMPY  ".center(80, "="))
    print("=" * 80)

    # 0. Reproducibility
    SEED = 42
    np.random.seed(SEED)
    ensure_directories()

    # 1. Dataset Loading
    print("\n[Step 1/6] Loading California Housing Dataset...")
    X_df, y_series = load_california_housing_data("data/california_housing.csv")
    full_df = pd.concat([X_df, y_series], axis=1)

    print(f"Dataset Shape: {full_df.shape[0]} samples, {full_df.shape[1]} columns")
    print(f"Features: {list(X_df.columns)}")
    print(f"Target: {y_series.name}")
    print("\nSummary Statistics of Features:")
    print(X_df.describe().T[["mean", "std", "min", "50%", "max"]].to_string())

    # 2. Exploratory Data Analysis & Visualizations
    print("\n[Step 2/6] Generating Exploratory Visualizations...")
    plot_feature_distributions(full_df, save_path="results/feature_distributions.png")
    plot_correlation_heatmap(full_df, save_path="results/correlation_heatmap.png")

    # 3. Data Splitting & Standardization (Leakage Prevention)
    print("\n[Step 3/6] Splitting Dataset and Preprocessing...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values, y_series.values, test_size=0.2, random_state=SEED
    )
    print(f"Training samples: {X_train_raw.shape[0]} (80%)")
    print(f"Testing samples:  {X_test_raw.shape[0]} (20%)")

    # Fit scaler strictly on training set to avoid data leakage
    scaler = StandardScalerScratch()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    print("Feature standardization complete (fit on X_train only).")

    # 4. Simple Linear Regression (Single Feature: MedInc -> MedHouseVal)
    print("\n[Step 4/6] Training Simple Linear Regression (Feature: 'MedInc')...")
    # MedInc is at column index 0
    medinc_idx = list(X_df.columns).index("MedInc")
    X_train_s = X_train[:, [medinc_idx]]
    X_test_s = X_test[:, [medinc_idx]]

    simple_lr = LinearRegressionScratch(learning_rate=0.02, iterations=1500)
    simple_lr.fit(X_train_s, y_train, verbose=False)
    y_pred_s = simple_lr.predict(X_test_s)

    simple_metrics = evaluate_model(y_test, y_pred_s)
    print_metrics_table(simple_metrics, title="Simple Linear Regression (MedInc)")
    print(f"Learned Weight: {simple_lr.weights[0]:.6f}")
    print(f"Learned Bias:   {simple_lr.bias:.6f}")

    plot_simple_linear_regression(
        X_train_s,
        y_train,
        X_test_s,
        y_test,
        simple_lr,
        feature_name="MedInc (Standardized)",
        target_name="MedHouseVal ($100k)",
        save_path="results/simple_linear_regression.png",
    )

    # 5. Multiple Linear Regression (All Features)
    print("\n[Step 5/6] Training Multiple Linear Regression (All Features)...")
    mult_lr = LinearRegressionScratch(learning_rate=0.05, iterations=3000)
    mult_lr.fit(X_train, y_train, verbose=True, print_every=500)

    y_pred_test = mult_lr.predict(X_test)
    custom_metrics = evaluate_model(y_test, y_pred_test)
    print_metrics_table(custom_metrics, title="NumPy Scratch Model Test Metrics")

    # Multiple Regression Plots
    plot_cost_convergence(mult_lr.cost_history, save_path="results/cost_convergence.png")
    plot_actual_vs_predicted(
        y_test,
        y_pred_test,
        model_name="Custom NumPy Multiple Linear Regression",
        save_path="results/actual_vs_predicted.png",
    )
    plot_residuals(y_test, y_pred_test, save_path="results/residual_plot.png")

    # 6. Scikit-learn Comparison
    print("\n[Step 6/6] Benchmarking Against Scikit-learn LinearRegression...")
    sklearn_lr = LinearRegression(fit_intercept=True)
    sklearn_lr.fit(X_train, y_train)

    y_pred_sklearn = sklearn_lr.predict(X_test)
    sklearn_metrics = evaluate_model(y_test, y_pred_sklearn)
    print_metrics_table(sklearn_metrics, title="Scikit-Learn (OLS) Test Metrics")

    # Comparison Table
    metrics_names = ["MSE", "RMSE", "MAE", "R2"]
    comparison_data = {
        "Metric": metrics_names,
        "Custom NumPy Scratch": [f"{custom_metrics[m]:.6f}" for m in metrics_names],
        "Scikit-learn OLS": [f"{sklearn_metrics[m]:.6f}" for m in metrics_names],
        "Absolute Difference": [f"{abs(custom_metrics[m] - sklearn_metrics[m]):.6e}" for m in metrics_names],
    }
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 80)
    print("  MODEL PERFORMANCE COMPARISON TABLE  ".center(80, "="))
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Coefficients Comparison Table
    coef_data = {
        "Feature": list(X_df.columns),
        "Custom NumPy Weight": mult_lr.weights,
        "Scikit-learn Coef": sklearn_lr.coef_,
        "Abs Difference": np.abs(mult_lr.weights - sklearn_lr.coef_),
    }
    coef_df = pd.DataFrame(coef_data)
    print("\n" + "=" * 80)
    print("  COEFFICIENTS COMPARISON TABLE  ".center(80, "="))
    print("=" * 80)
    print(coef_df.to_string(index=False))
    print(f"\nIntercept: Custom = {mult_lr.bias:.6f} | Sklearn = {sklearn_lr.intercept_:.6f} | Diff = {abs(mult_lr.bias - sklearn_lr.intercept_):.6e}")

    # Plot Model Comparison Bar Chart
    plot_model_comparison(
        custom_metrics,
        sklearn_metrics,
        save_path="results/model_comparison.png",
    )

    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!  ".center(80, "="))
    print("=" * 80)
    print("All plots generated and saved in 'results/':")
    for f in sorted(os.listdir("results")):
        print(f"  - results/{f}")

    return {
        "custom_metrics": custom_metrics,
        "sklearn_metrics": sklearn_metrics,
        "custom_weights": mult_lr.weights,
        "sklearn_weights": sklearn_lr.coef_,
        "custom_bias": mult_lr.bias,
        "sklearn_bias": sklearn_lr.intercept_,
        "features": list(X_df.columns),
    }


if __name__ == "__main__":
    run_pipeline()
