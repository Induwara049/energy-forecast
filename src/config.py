"""Project configuration settings for paths and constants."""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_FILE = DATA_DIR / "energy_data_set.csv"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Target column
TARGET_COLUMN = "Appliances"

# Datetime column
DATETIME_COLUMN = "date"