"""Utility functions for preprocessing the energy forecasting dataset."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.config import PROCESSED_DATA


def load_feature_engineered_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the feature-engineered dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded pandas DataFrame.
    """
    return pd.read_csv(file_path)


def parse_datetime_index(
    energy_df: pd.DataFrame,
    datetime_column: str = "date",
) -> pd.DataFrame:
    """
    Convert a datetime column to pandas datetime and set it as index.

    Args:
        energy_df: Input dataset.
        datetime_column: Name of the datetime column.

    Returns:
        DataFrame with datetime index.
    """
    df = energy_df.copy()
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df = df.sort_values(by=datetime_column)
    df = df.set_index(datetime_column)

    return df


def summarize_missing_values(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values for each column.

    Args:
        energy_df: Input dataset.

    Returns:
        DataFrame with missing counts and percentages.
    """
    missing_count = energy_df.isnull().sum()
    missing_percentage = (missing_count / len(energy_df)) * 100

    missing_summary_df = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        }
    ).sort_values(by="missing_count", ascending=False)

    return missing_summary_df


def chronological_train_test_split(
    energy_df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series dataset chronologically into train and test sets.

    Args:
        energy_df: Input dataset sorted by time.
        train_ratio: Fraction of data to use for training.

    Returns:
        Tuple of (train_df, test_df).
    """
    split_index = int(len(energy_df) * train_ratio)

    train_df = energy_df.iloc[:split_index].copy()
    test_df = energy_df.iloc[split_index:].copy()

    return train_df, test_df


def split_features_and_target(
    energy_df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into feature matrix and target vector.

    Args:
        energy_df: Input dataset.
        target_column: Name of the target column.

    Returns:
        Tuple of (X, y).
    """
    x_df = energy_df.drop(columns=[target_column])
    y_series = energy_df[target_column]

    return x_df, y_series


def scale_train_test_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    scaler_type: str = "standard",
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Scale training and testing feature sets using a fitted scaler on training data.

    Args:
        x_train: Training feature DataFrame.
        x_test: Testing feature DataFrame.
        scaler_type: Type of scaler to use ('standard' or 'minmax').

    Returns:
        Tuple of (x_train_scaled, x_test_scaled, fitted_scaler).
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be either 'standard' or 'minmax'.")

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, scaler


def create_sequences(
    x_data: np.ndarray,
    y_data: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequential samples for deep learning models such as LSTM.

    Args:
        x_data: Scaled feature array.
        y_data: Target array.
        sequence_length: Number of past time steps per sample.

    Returns:
        Tuple of (x_sequences, y_sequences).
    """
    x_sequences = []
    y_sequences = []

    for index in range(sequence_length, len(x_data)):
        x_sequences.append(x_data[index - sequence_length:index])
        y_sequences.append(y_data[index])

    return np.array(x_sequences), np.array(y_sequences)


def save_dataframe(
    energy_df: pd.DataFrame,
    file_name: str,
) -> None:
    """
    Save a DataFrame to CSV in the specified project-level directory.

    Args:
        energy_df: DataFrame to save.
        file_name: Output CSV file name.
    """
    # base_dir = BASE_DIR
    save_dir = PROCESSED_DATA

    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file_name

    energy_df.to_csv(file_path, index=True)