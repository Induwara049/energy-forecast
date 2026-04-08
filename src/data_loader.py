"""Utilities for loading and preparing the energy dataset."""

import pandas as pd


def load_energy_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the energy dataset from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the loaded dataset.
    """
    energy_df = pd.read_csv(file_path)
    return energy_df


def parse_datetime_column(
    energy_df: pd.DataFrame,
    datetime_column: str = "date",
    set_as_index: bool = True
) -> pd.DataFrame:
    """
    Convert the datetime column to pandas datetime format.

    Args:
        energy_df: Input dataset.
        datetime_column: Name of the datetime column.
        set_as_index: Whether to set the datetime column as the index.

    Returns:
        A DataFrame with parsed datetime values.
    """
    energy_df = energy_df.copy()
    energy_df[datetime_column] = pd.to_datetime(energy_df[datetime_column])

    if set_as_index:
        energy_df = energy_df.sort_values(by=datetime_column)
        energy_df = energy_df.set_index(datetime_column)

    return energy_df