"""
=============================================================================
Utility Functions — Data Loading & Preprocessing
=============================================================================
Reusable data loading, splitting, and preprocessing functions used across
all phases of the project.

Author: AI Research Assistant
Date: June 2026
=============================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(filepath, target_col='Target'):
    """
    Load a CSV dataset and separate features from target.

    Parameters
    ----------
    filepath : str
        Absolute path to the CSV file.
    target_col : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    feature_names : list
        List of feature column names.
    """
    df = pd.read_csv(filepath)
    feature_names = [col for col in df.columns if col != target_col]
    X = df[feature_names]
    y = df[target_col]
    return X, y, feature_names


def stratified_split(X, y, test_size=0.20, random_state=42):
    """
    Perform a stratified train-test split preserving class distribution.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float
        Proportion held out for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
        Split components.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def print_split_summary(y_train, y_test):
    """Print a formatted summary of the train-test split."""
    print(f"  Training: {len(y_train)} samples "
          f"(Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
    print(f"  Test:     {len(y_test)} samples "
          f"(Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)})")
