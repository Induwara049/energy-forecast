"""Utility functions for baseline and deep learning model development."""

import math
import joblib
from src.config import MODEL_RESULTS, MODEL_DIR, MODEL_FIG_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def train_linear_regression_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> LinearRegression:
    """
    Train a Linear Regression model.

    Args:
        x_train: Training feature array.
        y_train: Training target array.

    Returns:
        Trained LinearRegression model.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_random_forest_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Args:
        x_train: Training feature array.
        y_train: Training target array.
        n_estimators: Number of trees.
        random_state: Random seed.

    Returns:
        Trained RandomForestRegressor model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Evaluate regression predictions using MAE and RMSE.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing MAE and RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "MAE": mae,
        "RMSE": rmse,
    }


def create_lstm_model(
    input_shape: tuple,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
) -> Sequential:
    """
    Create an LSTM model for time-series regression.

    Args:
        input_shape: Shape of each input sequence (time_steps, num_features).
        lstm_units: Number of LSTM units.
        dropout_rate: Dropout rate after LSTM layer.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential(
        [
            LSTM(lstm_units, input_shape=input_shape),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


def get_early_stopping_callback(
    patience: int = 5,
) -> EarlyStopping:
    """
    Create an EarlyStopping callback.

    Args:
        patience: Number of epochs with no improvement before stopping.

    Returns:
        Configured EarlyStopping callback.
    """
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


def plot_training_history(history) -> None:
    """
    Plot training and validation loss history.

    Args:
        history: Keras History object from model training.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    save_model_plot("Training and Validation Loss")
    plt.show()


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    file_name: str,
) -> None:
    """
    Plot actual vs predicted target values.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        title: Plot title.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Appliances")
    plt.legend()
    plt.tight_layout()
    save_model_plot(file_name)
    plt.show()


def create_model_comparison_dataframe(
    linear_metrics: dict,
    random_forest_metrics: dict,
    lstm_metrics: dict,
) -> pd.DataFrame:
    """
    Create a comparison table for model evaluation metrics.

    Args:
        linear_metrics: Metrics for Linear Regression.
        random_forest_metrics: Metrics for Random Forest.
        lstm_metrics: Metrics for LSTM.

    Returns:
        DataFrame comparing model performance.
    """
    comparison_df = pd.DataFrame(
        [
            {
                "model": "Linear Regression",
                "MAE": linear_metrics["MAE"],
                "RMSE": linear_metrics["RMSE"],
            },
            {
                "model": "Random Forest",
                "MAE": random_forest_metrics["MAE"],
                "RMSE": random_forest_metrics["RMSE"],
            },
            {
                "model": "LSTM",
                "MAE": lstm_metrics["MAE"],
                "RMSE": lstm_metrics["RMSE"],
            },
        ]
    )

    return comparison_df


def save_model_comparison_dataframe(
    comparison_df: pd.DataFrame,
    file_name: str,
) -> None:
    """
    Save the model comparison DataFrame to CSV in a project-level directory.

    Args:
        comparison_df: DataFrame to save.
        file_name: Output CSV file name.
    """
    save_dir = MODEL_RESULTS

    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file_name

    comparison_df.to_csv(file_path, index=False)




def save_model(model, file_name: str) -> None:
    """
    Save a scikit-learn model to the project-level models directory.

    Args:
        model: Trained scikit-learn model.
        file_name: File name to save the model.
    """
    save_dir = MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / file_name
    joblib.dump(model, file_path)



def save_model_plot(file_name: str) -> None:
    """
    Save the current matplotlib figure to the project-level model plots directory.

    Args:
        file_name: File name to save the plot.
        directory: Output directory relative to project root.
    """
    save_dir = MODEL_FIG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / file_name
    plt.savefig(file_path, dpi=300, bbox_inches="tight")