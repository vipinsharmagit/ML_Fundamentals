"""
Linear Regression From Scratch Using NumPy
==========================================
This module provides a production-grade, object-oriented implementation of
Linear Regression optimized with Batch Gradient Descent using pure NumPy.
"""

from typing import List, Optional, Union
import numpy as np


class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using NumPy.

    Optimizes weights and bias using Batch Gradient Descent to minimize
    the Mean Squared Error (MSE) objective function. Supports both univariate
    (single feature) and multivariate (multiple features) regression.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size used for gradient descent parameter updates (alpha).
    iterations : int, default=1000
        Total number of gradient descent iterations (epochs).

    Attributes
    ----------
    weights : np.ndarray of shape (n_features,)
        Learned feature weights after model fitting.
    bias : float
        Learned intercept (bias term) after model fitting.
    cost_history : List[float]
        Mean squared error cost recorded after each training iteration.
    n_features_in_ : int
        Number of features seen during the fit call.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if iterations <= 0:
            raise ValueError(f"iterations must be a positive integer, got {iterations}")

        self.learning_rate: float = learning_rate
        self.iterations: int = iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.cost_history: List[float] = []
        self.n_features_in_: Optional[int] = None

    def _validate_and_format_input(
        self, X: Union[np.ndarray, list], y: Optional[Union[np.ndarray, list]] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and format input arrays for consistency and numerical stability.
        Converts 1D feature arrays into 2D column matrices (m, 1).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        elif X_arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array for X, got shape {X_arr.shape}")

        y_arr = None
        if y is not None:
            y_arr = np.asarray(y, dtype=np.float64)
            if y_arr.ndim == 2:
                if y_arr.shape[1] == 1:
                    y_arr = y_arr.ravel()
                else:
                    raise ValueError(f"Target y must be 1D, got shape {y_arr.shape}")
            elif y_arr.ndim != 1:
                raise ValueError(f"Target y must be 1D, got shape {y_arr.shape}")

            if X_arr.shape[0] != y_arr.shape[0]:
                raise ValueError(
                    f"Sample count mismatch: X has {X_arr.shape[0]} rows, y has {y_arr.shape[0]} elements."
                )

        return X_arr, y_arr

    def hypothesis(self, X: np.ndarray) -> np.ndarray:
        """
        Hypothesis function:
            h(X) = X @ weights + bias

        Parameters
        ----------
        X : np.ndarray of shape (m, n)
            Input feature matrix.

        Returns
        -------
        predictions : np.ndarray of shape (m,)
            Predicted target values.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' before evaluating hypothesis.")
        return X @ self.weights + self.bias

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) objective function:
            J(w, b) = (1 / (2 * m)) * sum((h(X) - y) ** 2)

        Parameters
        ----------
        X : np.ndarray of shape (m, n)
            Feature matrix.
        y : np.ndarray of shape (m,)
            True continuous targets.

        Returns
        -------
        cost : float
            Computed Mean Squared Error cost.
        """
        m = len(y)
        if m == 0:
            return 0.0
        predictions = self.hypothesis(X)
        cost = (1.0 / (2.0 * m)) * np.sum((predictions - y) ** 2)
        return float(cost)

    def fit(
        self,
        X: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        verbose: bool = False,
        print_every: int = 100,
    ) -> "LinearRegressionScratch":
        """
        Fit the linear model via Batch Gradient Descent.

        Parameters
        ----------
        X : array-like of shape (m, n) or (m,)
            Training feature matrix.
        y : array-like of shape (m,)
            Training target vector.
        verbose : bool, default=False
            If True, prints training loss progress at intervals.
        print_every : int, default=100
            Number of iterations between progress logs if verbose is True.

        Returns
        -------
        self : LinearRegressionScratch
            Fitted estimator instance.
        """
        X_mat, y_vec = self._validate_and_format_input(X, y)
        m, n = X_mat.shape

        self.n_features_in_ = n
        # Initialize parameters
        self.weights = np.zeros(n, dtype=np.float64)
        self.bias = 0.0
        self.cost_history = []

        if verbose:
            initial_cost = self.compute_cost(X_mat, y_vec)
            print(f"Starting Gradient Descent | Samples: {m}, Features: {n}")
            print(f"Iteration      0/{self.iterations} | Initial Cost J(w,b): {initial_cost:.6f}")

        # Batch Gradient Descent Optimization Loop
        for iteration in range(1, self.iterations + 1):
            # 1. Forward propagation (Hypothesis evaluation)
            predictions = self.hypothesis(X_mat)
            errors = predictions - y_vec

            # 2. Record cost before parameter update
            cost = (1.0 / (2.0 * m)) * np.sum(errors ** 2)
            self.cost_history.append(float(cost))

            # Numerical stability guard against diverging gradients
            if np.isnan(cost) or np.isinf(cost):
                raise FloatingPointError(
                    f"Overflow/NaN detected at iteration {iteration}. "
                    "Learning rate may be too large. Try standardizing features or reducing alpha."
                )

            # 3. Compute gradients via matrix operations
            # dw = (1/m) * X.T @ (predictions - y)
            # db = (1/m) * sum(predictions - y)
            dw = (1.0 / m) * (X_mat.T @ errors)
            db = (1.0 / m) * np.sum(errors)

            # 4. Update parameters using gradient descent step
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Log progress if verbose
            if verbose and (iteration % print_every == 0 or iteration == self.iterations):
                print(f"Iteration {iteration:6d}/{self.iterations} | Cost J(w,b): {cost:.6f}")

        return self

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Generate continuous predictions for given input features.

        Parameters
        ----------
        X : array-like of shape (m, n) or (m,)
            Input features.

        Returns
        -------
        predictions : np.ndarray of shape (m,)
            Predicted target values.
        """
        X_mat, _ = self._validate_and_format_input(X)
        if self.n_features_in_ is not None and X_mat.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: model was fitted with {self.n_features_in_} features, "
                f"but received input with {X_mat.shape[1]} features."
            )
        return self.hypothesis(X_mat)

    def __repr__(self) -> str:
        status = "Fitted" if self.weights is not None else "Unfitted"
        return (
            f"LinearRegressionScratch(learning_rate={self.learning_rate}, "
            f"iterations={self.iterations}, status={status})"
        )
