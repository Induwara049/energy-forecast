"""Utility functions for exploratory data analysis."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.config import EDA_FIG_DIR

def save_plot(file_name: str) -> None:
    """
    Save plot to project-level outputs/eda_plots directory.
    """

    output_dir = EDA_FIG_DIR
    os.makedirs(output_dir, exist_ok=True)

    file_path = output_dir / file_name

    plt.savefig(file_path, dpi=300, bbox_inches="tight")


def summarize_dataset(energy_df: pd.DataFrame) -> None:
    """
    Print a concise summary of the dataset.

    Args:
        energy_df: Input dataset.
    """
    print("Shape:", energy_df.shape)
    print("\nData types:\n", energy_df.dtypes)
    print("\nFirst 5 rows:\n", energy_df.head())


def summarize_missing_values(energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values by column.

    Args:
        energy_df: Input dataset.

    Returns:
        DataFrame containing missing count and percentage.
    """
    missing_count = energy_df.isnull().sum()
    missing_percentage = (missing_count / len(energy_df)) * 100

    missing_summary_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    }).sort_values(by="missing_count", ascending=False)

    return missing_summary_df


def plot_target_over_time(
    energy_df: pd.DataFrame,
    target_column: str,
    file_name: str,
    figsize: tuple = (16, 5),
) -> None:
    """
    Plot the target variable over time and save it.

    Args:
        energy_df: Input dataset with datetime index.
        target_column: Name of the target column.
        file_name: File name to save the plot.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    plt.plot(energy_df.index, energy_df[target_column])
    plt.title(f"{target_column} Over Time")
    plt.xlabel("Time")
    plt.ylabel(target_column)
    plt.tight_layout()

    save_plot(file_name)
    plt.show()


def plot_correlation_heatmap(
    energy_df: pd.DataFrame,
    file_name: str,
    figsize: tuple = (16, 10),
) -> None:
    """
    Plot a correlation heatmap for numeric columns and save it.

    Args:
        energy_df: Input dataset.
        file_name: File name to save the plot.
        figsize: Figure size.
    """
    correlation_matrix = energy_df.corr(numeric_only=True)

    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    save_plot(file_name)
    plt.show()


def plot_boxplots_for_selected_columns(
    energy_df: pd.DataFrame,
    selected_columns: list[str],
    file_name: str,
    figsize: tuple = (14, 6),
) -> None:
    """
    Plot boxplots for selected columns and save them.

    Args:
        energy_df: Input dataset.
        selected_columns: Columns to visualize.
        file_name: File name to save the plot.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    energy_df[selected_columns].boxplot()
    plt.title("Boxplots for Selected Features")
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_plot(file_name)
    plt.show()


def plot_average_energy_by_hour(
    energy_df: pd.DataFrame,
    target_column: str,
    file_name: str,
    figsize: tuple = (10, 5),
) -> None:
    """
    Plot average energy consumption by hour and save it.

    Args:
        energy_df: Input dataset with datetime index.
        target_column: Target column name.
        file_name: File name to save the plot.
        figsize: Figure size.
    """
    # Extract hour
    hourly_series = energy_df.index.hour

    # Compute average
    hourly_avg = energy_df.groupby(hourly_series)[target_column].mean()

    plt.figure(figsize=figsize)
    plt.plot(hourly_avg.index, hourly_avg.values, marker="o")
    plt.title("Average Energy Consumption by Hour")
    plt.xlabel("Hour")
    plt.ylabel(target_column)
    plt.xticks(range(0, 24))
    plt.tight_layout()

    save_plot(file_name)
    plt.show()


def plot_target_distribution(
    energy_df: pd.DataFrame,
    target_column: str,
    file_name: str,
    bins: int = 50,
    figsize: tuple = (10, 5),
) -> None:
    """
    Plot distribution of the target variable and save it.

    Args:
        energy_df: Input dataset.
        target_column: Target column name.
        file_name: File name to save the plot.
        bins: Number of histogram bins.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    energy_df[target_column].hist(bins=bins)
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Frequency")
    plt.tight_layout()

    save_plot(file_name)
    plt.show()   