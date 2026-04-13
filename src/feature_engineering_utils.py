"""Utility functions for feature engineering on the energy dataset."""

import os
import pandas as pd
from typing import List
from src.config import PROCESSED_DATA


def create_time_features(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the datetime index.

    Args:
        energy_df: Input dataset with a datetime index.

    Returns:
        DataFrame with additional time-based features.
    """
    df = energy_df.copy()

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_month"] = df.index.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def create_lag_features(
    energy_df: pd.DataFrame,
    target_column: str,
    lag_steps: list[int],
) -> pd.DataFrame:
    """
    Create lagged versions of the target column.

    Args:
        energy_df: Input dataset.
        target_column: Target column name.
        lag_steps: List of lag steps to create.

    Returns:
        DataFrame with lag features added.
    """
    df = energy_df.copy()

    for lag in lag_steps:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)

    return df


def create_rolling_features(
    energy_df: pd.DataFrame,
    target_column: str,
    window_sizes: list[int],
) -> pd.DataFrame:
    """
    Create rolling mean features for the target column.

    Args:
        energy_df: Input dataset.
        target_column: Target column name.
        window_sizes: List of rolling window sizes.

    Returns:
        DataFrame with rolling mean features added.
    """
    df = energy_df.copy()

    for window in window_sizes:
        df[f"{target_column}_rolling_mean_{window}"] = (
            df[target_column].rolling(window=window).mean()
        )

    return df


def create_interaction_features(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between selected environmental variables.

    Args:
        energy_df: Input dataset.

    Returns:
        DataFrame with interaction features added.
    """
    df = energy_df.copy()

    if {"T_out", "RH_out"}.issubset(df.columns):
        df["T_out_RH_out_interaction"] = df["T_out"] * df["RH_out"]

    if {"T1", "RH_1"}.issubset(df.columns):
        df["T1_RH_1_interaction"] = df["T1"] * df["RH_1"]

    if {"T2", "RH_2"}.issubset(df.columns):
        df["T2_RH_2_interaction"] = df["T2"] * df["RH_2"]

    return df

def clip_outliers_iqr(
    energy_df: pd.DataFrame,
    columns: list[str],
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Clip outliers in selected numeric columns using the IQR rule.

    Args:
        energy_df: Input dataset.
        columns: List of columns to clip.
        iqr_multiplier: IQR multiplier for outlier boundaries.

    Returns:
        DataFrame with clipped values.
    """
    df = energy_df.copy()

    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df

def drop_feature_engineering_nulls(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing null values created by lag and rolling operations.

    Args:
        energy_df: Input dataset.

    Returns:
        Cleaned DataFrame with null rows removed.
    """
    df = energy_df.copy()
    df = df.dropna()

    return df


def save_dataframe(
    energy_df: pd.DataFrame,
    file_name: str,
) -> None:
    """
    Save a DataFrame to CSV in the specified directory.

    Args:
        energy_df: DataFrame to save.
        file_name: Output CSV file name.
        directory: Output directory path.
    """
    os.makedirs(PROCESSED_DATA, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA, file_name)
    energy_df.to_csv(file_path, index=True)


def select_columns(
    df: pd.DataFrame,
    columns: List[str],
    strict: bool = True
) -> pd.DataFrame:
    """
    Select specific columns from a DataFrame.

    Args:
        df: Input DataFrame.
        columns: List of column names to select.
        strict: 
            - True → raise error if any column is missing
            - False → ignore missing columns

    Returns:
        DataFrame with selected columns.
    """

    if strict:
        # Ensure all requested columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        return df[columns]

    else:
        # Select only available columns
        available_cols = [col for col in columns if col in df.columns]
        return df[available_cols]