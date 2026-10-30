"""
Visualization Module for Linear Regression
==========================================
Produces publication-quality figures using Matplotlib and Seaborn for:
- Feature distributions
- Correlation heatmaps
- Gradient descent cost convergence
- Actual vs. Predicted regression plots
- Residual diagnostics
- Univariate regression line fitting
- Benchmark model comparisons
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_plotting_style() -> None:
    """Set global consistent plotting aesthetics."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot histogram with KDE for numerical features in the dataset.
    """
    set_plotting_style()
    if features is None:
        features = list(df.columns)

    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(features):
        ax = axes[idx]
        sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4", bins=30, edgecolor="black", alpha=0.6)
        ax.set_title(f"Distribution: {col}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density / Count", fontsize=10)

    # Hide any unused axes
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Exploratory Data Analysis: Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved feature distributions plot to: {save_path}")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot correlation matrix heatmap of dataset features.
    """
    set_plotting_style()
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
    )
    plt.title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved correlation heatmap to: {save_path}")
    plt.close()


def plot_cost_convergence(
    cost_history: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot Mean Squared Error cost J(w,b) as a function of gradient descent iterations.
    """
    set_plotting_style()
    plt.figure(figsize=(9, 5.5))
    iterations = range(1, len(cost_history) + 1)

    plt.plot(
        iterations,
        cost_history,
        color="#0066cc",
        linewidth=2.2,
        label=r"Training Cost $J(\mathbf{w}, b)$",
    )

    initial_cost = cost_history[0]
    final_cost = cost_history[-1]

    # Annotate initial and converged costs
    plt.scatter([1], [initial_cost], color="#d62728", s=60, zorder=5)
    plt.annotate(
        f"Initial: {initial_cost:.4f}",
        xy=(1, initial_cost),
        xytext=(len(cost_history) * 0.05, initial_cost * 0.95),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        fontsize=10,
        fontweight="bold",
        color="#d62728",
    )

    plt.scatter([len(cost_history)], [final_cost], color="#2ca02c", s=60, zorder=5)
    plt.annotate(
        f"Final: {final_cost:.4f}",
        xy=(len(cost_history), final_cost),
        xytext=(len(cost_history) * 0.75, final_cost + (initial_cost - final_cost) * 0.15),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        fontsize=10,
        fontweight="bold",
        color="#2ca02c",
    )

    plt.title("Gradient Descent Optimization: Cost Function Convergence", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Iteration (Epoch)", fontsize=11, fontweight="bold")
    plt.ylabel(r"Cost $J(\mathbf{w}, b) = \frac{1}{2m} \sum (h - y)^2$", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved cost convergence plot to: {save_path}")
    plt.close()


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Linear Regression (NumPy From Scratch)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot actual vs predicted scatter plot with reference line y = x.
    """
    set_plotting_style()
    plt.figure(figsize=(8, 6.5))

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    # Scatter points with slight transparency to show density
    plt.scatter(y_t, y_p, color="#2b5c8f", alpha=0.35, edgecolors="none", s=22, label="Predictions")

    # Ideal prediction line y = x
    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))
    margin = (max_val - min_val) * 0.05
    line_min = min_val - margin
    line_max = max_val + margin

    plt.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label="Ideal Reference ($y = \hat{y}$)",
    )

    # Compute manual R2 & RMSE for the subtitle/annotation
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))

    plt.title(f"Actual vs. Predicted Values\n{model_name}", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Actual Ground Truth ($y$)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Values ($\hat{y}$)", fontsize=11, fontweight="bold")
    plt.xlim(line_min, line_max)
    plt.ylim(line_min, line_max)

    # Add performance text box
    stats_text = f"$R^2 = {r2:.4f}$\n$\\mathrm{{RMSE}} = {rmse:.4f}$"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    plt.gca().text(0.05, 0.92, stats_text, transform=plt.gca().transAxes, fontsize=11, verticalalignment="top", bbox=props)

    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved actual vs predicted plot to: {save_path}")
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Diagnostic residual plots:
    Left: Residuals vs. Predicted values (homoscedasticity check).
    Right: Residual Distribution histogram + KDE (normality check).
    """
    set_plotting_style()
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    residuals = y_t - y_p

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_p, residuals, color="#3470a3", alpha=0.4, s=20, edgecolors="none")
    ax1.axhline(0, color="#d62728", linestyle="--", linewidth=1.8, label="Zero Error Line")
    ax1.set_title("Residuals vs. Fitted Values", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Fitted Values ($\hat{y}$)", fontsize=11)
    ax1.set_ylabel("Residuals ($y - \hat{y}$)", fontsize=11)
    ax1.legend(frameon=True, facecolor="white")

    # Subplot 2: Histogram of Residuals
    ax2 = axes[1]
    sns.histplot(residuals, kde=True, ax=ax2, color="#2ca02c", bins=40, edgecolor="black", alpha=0.6)
    ax2.axvline(0, color="#d62728", linestyle="--", linewidth=1.8, label="Zero Mean")
    ax2.set_title("Residual Distribution (Error Normality)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Residual Value ($y - \hat{y}$)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.legend(frameon=True, facecolor="white")

    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    res_box = f"Residual $\\mu = {mean_res:.4f}$\nResidual $\\sigma = {std_res:.4f}$"
    props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    ax2.text(0.05, 0.92, res_box, transform=ax2.transAxes, fontsize=10, verticalalignment="top", bbox=props)

    plt.suptitle("Residual Diagnostics & Error Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved residual plot to: {save_path}")
    plt.close()


def plot_simple_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: any,
    feature_name: str = "Median Income (Standardized)",
    target_name: str = "Median House Value ($100k)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot univariate simple linear regression line over scatter data.
    """
    set_plotting_style()
    plt.figure(figsize=(9, 6))

    x_tr = np.asarray(X_train).ravel()
    y_tr = np.asarray(y_train).ravel()
    x_te = np.asarray(X_test).ravel()
    y_te = np.asarray(y_test).ravel()

    # Scatter points for train and test
    plt.scatter(x_tr, y_tr, color="#4a90e2", alpha=0.25, s=20, label="Train Data", edgecolors="none")
    plt.scatter(x_te, y_te, color="#f5a623", alpha=0.35, s=24, label="Test Data", edgecolors="none")

    # Generate regression line across domain
    x_min = min(float(np.min(x_tr)), float(np.min(x_te)))
    x_max = max(float(np.max(x_tr)), float(np.max(x_te)))
    x_line = np.linspace(x_min, x_max, 300).reshape(-1, 1)
    y_line = model.predict(x_line)

    w_val = float(model.weights[0])
    b_val = float(model.bias)
    line_label = f"Regression Line: $\hat{{y}} = {w_val:.3f}x + {b_val:.3f}$"

    plt.plot(x_line.ravel(), y_line.ravel(), color="#d0021b", linewidth=2.5, label=line_label)

    plt.title(f"Simple Linear Regression Fit: {feature_name} vs. {target_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel(feature_name, fontsize=11, fontweight="bold")
    plt.ylabel(target_name, fontsize=11, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved simple linear regression plot to: {save_path}")
    plt.close()


def plot_model_comparison(
    custom_metrics: Dict[str, float],
    sklearn_metrics: Dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    """
    Create side-by-side bar chart comparison between Custom Scratch Model and Scikit-learn.
    """
    set_plotting_style()
    metric_names = ["MSE", "RMSE", "MAE", "R2"]
    custom_vals = [custom_metrics[m] for m in metric_names]
    sklearn_vals = [sklearn_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - bar_width / 2,
        custom_vals,
        bar_width,
        label="NumPy From Scratch",
        color="#2b7bba",
        edgecolor="black",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x + bar_width / 2,
        sklearn_vals,
        bar_width,
        label="Scikit-Learn (OLS SVD)",
        color="#f39c12",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Test Set Performance: NumPy From Scratch vs. Scikit-learn", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m} (Lower is better)" if m != "R2" else f"{m} (Higher is better)" for m in metric_names], fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.set_ylim(0, max(max(custom_vals), max(sklearn_vals)) * 1.25)

    # Attach value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved model comparison plot to: {save_path}")
    plt.close()
