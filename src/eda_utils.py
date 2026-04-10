"""Utility functions for exploratory data analysis."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def save_plot(file_name: str, directory: str = "outputs/eda_plots") -> None:
    """
    Save the current matplotlib figure to a specified directory.

    Args:
        file_name: Name of the file to save (e.g., 'plot.png').
        directory: Directory where the plot should be stored.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)
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
    target_column: str = "Appliances",
    figsize: tuple = (16, 5),
    file_name: str | None = None,
) -> None:
    """
    Plot the target variable over time.

    Args:
        energy_df: Input dataset with datetime index.
        target_column: Name of the target column.
        figsize: Figure size.
        file_name: Optional file name for saving the plot.
    """
    plt.figure(figsize=figsize)
    plt.plot(energy_df.index, energy_df[target_column])
    plt.title(f"{target_column} Over Time")
    plt.xlabel("Time")
    plt.ylabel(target_column)
    plt.tight_layout()

    if file_name is not None:
        save_plot(file_name)

    plt.show()


def plot_correlation_heatmap(
    energy_df: pd.DataFrame,
    figsize: tuple = (16, 10),
    file_name: str | None = None,
) -> None:
    """
    Plot a correlation heatmap for numeric columns.

    Args:
        energy_df: Input dataset.
        figsize: Figure size.
        file_name: Optional file name for saving the plot.
    """
    correlation_matrix = energy_df.corr(numeric_only=True)

    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    if file_name is not None:
        save_plot(file_name)

    plt.show()


def plot_boxplots_for_selected_columns(
    energy_df: pd.DataFrame,
    selected_columns: list[str],
    figsize: tuple = (14, 6),
    file_name: str | None = None,
) -> None:
    """
    Plot boxplots for selected columns.

    Args:
        energy_df: Input dataset.
        selected_columns: Columns to visualize.
        figsize: Figure size.
        file_name: Optional file name for saving the plot.
    """
    plt.figure(figsize=figsize)
    energy_df[selected_columns].boxplot()
    plt.title("Boxplots for Selected Features")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if file_name is not None:
        save_plot(file_name)

    plt.show()