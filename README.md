# 🔋 Appliance Energy Consumption Prediction

## 📌 Project Overview

This project focuses on predicting appliance energy consumption using a multivariate time-series dataset. The dataset consists of environmental, temporal, and energy-related features recorded at 10-minute intervals over several months.

The objective of this project is to analyze energy consumption patterns and build predictive models that can accurately estimate appliance usage. Both traditional machine learning models and deep learning approaches were explored to compare their effectiveness for structured time-series data.

---

## 📊 Dataset Description

- Approximately 20,000 records  
- Time interval: 10 minutes  
- Target variable: **Appliances (energy consumption in Wh)**  

### Key Features
- `Lights` – Energy consumption of lighting  
- `T1–T9` – Indoor temperatures  
- `RH_1–RH_9` – Indoor humidity levels  
- `T_out`, `RH_out` – Outdoor conditions  
- `Windspeed`, `Visibility`, `Pressure`  
- Time-based features derived from timestamps  

---

## ⚙️ Project Workflow

### Exploratory Data Analysis (EDA)
- Time-series visualization  
- Correlation heatmap analysis  
- Outlier detection using boxplots  
- Distribution analysis  

### Feature Engineering
- Time-based features (hour, day, etc.)  
- Lag features (previous values)  
- Rolling averages  
- Interaction features (e.g., temperature × humidity)  
- Feature selection based on correlation  

### Data Preprocessing
- Missing value handling (introduced during feature engineering)  
- Outlier treatment using IQR clipping  
- Chronological train-test split (80%-20%)  
- Feature scaling using StandardScaler  

### Model Development
- Linear Regression (baseline model)  
- Random Forest (ensemble model)  
- LSTM (deep learning model for sequence learning)  

### Model Evaluation
Models were evaluated using:
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

#### Results

| Model | MAE | RMSE |
|------|-----|------|
| Linear Regression | 11.61 | 18.01 |
| Random Forest | **11.53** | **17.73** |
| LSTM | 14.93 | 22.72 |

👉 Random Forest achieved the best performance.

---

### Model Optimization
- Random Forest hyperparameter tuning  
- LSTM improvements (sequence length, epochs, dropout)  
- Early stopping for better generalization  

---

## 🧠 Key Insights

- Feature engineering significantly improved model performance  
- Random Forest outperformed LSTM for structured datasets  
- LSTM struggled with sharp spikes and extreme values  
- Increased model complexity does not always guarantee better results  

---

## 📁 Project Structure

```
├── dataset/
│ ├── train_feature_engineered_data.csv
│ ├── test_feature_engineered_data.csv
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_preprocessing.ipynb
│ ├── 04_model_development.ipynb
│ ├── 05_model_optimization.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── eda_utils.py
│ ├── feature_engineering_utils.py
│ ├── preprocessing_utils.py
│ ├── model_utils.py
│
├── outputs/
│ ├── eda_plots/
│ ├── model_plots/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Guide

To run this project locally, ensure that Python (version 3.9 or higher) and pip are installed on your system. It is recommended to use a virtual environment to manage dependencies and avoid conflicts.

1) A virtual environment can be created using conda:

```bash
conda create -n energy_env python=3.9
conda activate energy_env
```
2) Once the environment is activated, install the required dependencies:
```
pip install -r requirements.txt
```
3) Ensure that the dataset files are placed inside the dataset/ directory and that the project structure is maintained as shown above.

4) The project is executed using Jupyter notebooks. Launch Jupyter Notebook:
```
jupyter notebook
```
5) Run the notebooks in the following order

01_eda.ipynb
02_feature_engineering.ipynb
03_preprocessing.ipynb
04_model_development.ipynb
05_model_optimization.ipynb