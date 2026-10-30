"""
Preprocessing Utilities From Scratch Using NumPy
=================================================
Includes a pure NumPy implementation of StandardScaler to ensure proper
feature scaling without data leakage between train and test splits.
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


class StandardScalerScratch:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the training samples.

    Formula:
        z = (x - u) / s
    where u is the mean of the training samples and s is the standard deviation.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    scale_ : np.ndarray of shape (n_features,)
        Per-feature relative scaling (standard deviation) of the data.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "StandardScalerScratch":
        """
        Compute the mean and std to be used for later scaling.
        Only fit on training data to prevent data leakage!

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns
        -------
        self : StandardScalerScratch
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)

        # Handle constant features (std == 0) to avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns
        -------
        X_scaled : np.ndarray
            Standardized data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerScratch is not fitted yet. Call 'fit' before 'transform'.")

        X_arr = np.asarray(X, dtype=np.float64)
        is_1d = X_arr.ndim == 1
        if is_1d:
            X_arr = X_arr.reshape(-1, 1)

        X_scaled = (X_arr - self.mean_) / self.scale_
        return X_scaled.ravel() if (is_1d and X_scaled.shape[1] == 1) else X_scaled

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_scaled : np.ndarray
            Standardized training data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Scale back the data to the original representation.

        Parameters
        ----------
        X_scaled : array-like of shape (n_samples, n_features)
            Standardized data.

        Returns
        -------
        X_orig : np.ndarray
            Original data scale.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerScratch is not fitted yet.")
        X_arr = np.asarray(X_scaled, dtype=np.float64)
        return (X_arr * self.scale_) + self.mean_


def load_california_housing_data(
    data_path: str = "data/california_housing.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load California Housing dataset using Pandas.
    If the clean CSV does not exist locally, process from data/housing_raw.csv.

    Returns
    -------
    X_df : pd.DataFrame
        Features dataframe (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude).
    y_series : pd.Series
        Target series (MedHouseVal in $100k).
    """
    if not os.path.exists(data_path):
        raw_path = "data/housing_raw.csv"
        if os.path.exists(raw_path):
            print(f"Transforming raw housing data from: {raw_path}")
            raw_df = pd.read_csv(raw_path)
            # Impute missing total_bedrooms before calculating ratios
            raw_df["total_bedrooms"] = raw_df["total_bedrooms"].fillna(raw_df["total_bedrooms"].median())

            df = pd.DataFrame()
            df["MedInc"] = raw_df["median_income"]
            df["HouseAge"] = raw_df["housing_median_age"]
            df["AveRooms"] = raw_df["total_rooms"] / raw_df["households"]
            df["AveBedrms"] = raw_df["total_bedrooms"] / raw_df["households"]
            df["Population"] = raw_df["population"]
            df["AveOccup"] = raw_df["population"] / raw_df["households"]
            df["Latitude"] = raw_df["latitude"]
            df["Longitude"] = raw_df["longitude"]
            df["MedHouseVal"] = raw_df["median_house_value"] / 100000.0

            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
            print(f"Saved processed dataset to: {data_path}")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Check and handle missing values
    missing_count = int(df.isnull().sum().sum())
    if missing_count > 0:
        print(f"Handling {missing_count} missing values using median imputation...")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("Data integrity check passed: 0 missing values detected.")

    target_col = "MedHouseVal"
    feature_cols = [c for c in df.columns if c != target_col]
    X_df = df[feature_cols]
    y_series = df[target_col]

    return X_df, y_series
