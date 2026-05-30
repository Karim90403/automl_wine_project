# Automated ML Pipeline for Wine Quality Prediction - Автоматизация ML - Музафаров Карим Ринатович

## Project Overview

This project implements an automated machine learning (AutoML) pipeline for predicting wine quality using the Wine Quality dataset from the UCI Machine Learning Repository. The pipeline automates the entire ML workflow from data ingestion to model deployment.

**Dataset:** [Wine Quality (UCI)](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Final Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.843 |
| Precision | 0.583 |
| Recall | 0.609 |
| F1 Score | 0.596 |
| ROC AUC | 0.869 |

**Best Model:** Random Forest
- n_estimators: 147
- max_depth: 21
- min_samples_split: 12
- min_samples_leaf: 4
- max_features: log2

**Features Used:** 20 features (11 original + 9 engineered)

## Features

- **Automated Data Download**: Downloads red and white wine quality datasets from UCI
- **Data Preprocessing**: Handles missing values, duplicates, and feature engineering
- **Automated Hyperparameter Tuning**: Uses Optuna for efficient hyperparameter optimization
- **Multiple Model Support**: Compares Logistic Regression, Random Forest, Extra Trees, and Gradient Boosting
- **Comprehensive Evaluation**: Generates metrics, confusion matrix, ROC curve, and feature importance
- **Model Persistence**: Saves best model using joblib
- **Experiment Tracking**: Optional MLflow integration for experiment tracking
- **Docker Support**: Containerized deployment with Docker Compose

## Project Structure

```
automl_wine_project/
├── data/
│   ├── raw/                 # Raw downloaded datasets
│   └── processed/           # Processed datasets
├── models/                  # Saved models
├── reports/
│   ├── figures/             # Visualization plots
│   └── tables/              # Metrics and results
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration parameters
│   ├── download_data.py     # Data download functionality
│   ├── etl.py               # Data preprocessing pipeline
│   ├── train.py             # Training with Optuna optimization
│   ├── evaluate.py          # Model evaluation
│   ├── predict.py           # Prediction interface
│   └── monitor.py           # Resource monitoring
├── tests/
│   ├── test_etl.py          # ETL tests
│   └── test_train.py        # Training tests
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pytest.ini
└── requirements.txt
```

## Installation

```bash
# Install dependencies
make install
# or
pip install -r requirements.txt
```

## Usage

### 1. Download Data

```bash
make download
# or
python -m src.download_data
```

### 2. Train Model

```bash
make train
# or
python -m src.train
```

### 3. Evaluate Model

```bash
make evaluate
# or
python -m src.evaluate
```

### 4. Run Tests

```bash
make test
# or
pytest
```

### 5. Docker Deployment

```bash
make docker-build
make docker-run
# or
docker compose build
docker compose up
```

## Model Details

### Task Formulation

The project converts the wine quality prediction into a **binary classification** problem:
- **Class 0 (Low Quality)**: Wines with quality score < 7
- **Class 1 (High Quality)**: Wines with quality score >= 7

### Models Compared

| Model | Key Hyperparameters |
|-------|---------------------|
| Logistic Regression | C (regularization) |
| Random Forest | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| Extra Trees | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| Gradient Boosting | n_estimators, learning_rate, max_depth, subsample |

### Hyperparameter Optimization

Uses **Optuna** with Tree-structured Parzen Estimator (TPE) sampler to maximize F1-score on the validation set.

## Dataset Description

The Wine Quality dataset contains the following features:

| Feature | Description |
|---------|-------------|
| fixed acidity | Total acidity (grams of tartaric acid per dm³) |
| volatile acidity | Volatile acidity (grams of acetic acid per dm³) |
| citric acid | Citric acid (grams per dm³) |
| residual sugar | Residual sugar (grams per dm³) |
| chlorides | Chlorides (grams of NaCl per dm³) |
| free sulfur dioxide | Free SO₂ (mg/dm³) |
| total sulfur dioxide | Total SO₂ (mg/dm³) |
| density | Density (g/cm³) |
| pH | pH value |
| sulphates | Sulphates (potassium sulfate) |
| alcohol | Alcohol content (%) |
| quality | Score between 0 and 10 |
| wine_type | 0 = red, 1 = white |

## Output Files

After training, the following files are generated:

- `models/best_model.joblib` - Trained model
- `reports/metrics.json` - Performance metrics and best parameters
- `reports/figures/confusion_matrix.png` - Confusion matrix visualization
- `reports/figures/roc_curve.png` - ROC curve visualization
- `reports/figures/metrics.png` - Bar chart of metrics
- `reports/figures/target_distribution.png` - Target class distribution
- `reports/figures/feature_importance.png` - Feature importance plot
- `reports/tables/optuna_trials.csv` - Optuna trial results

## Example Prediction

```python
from src.predict import predict

sample_wine = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "wine_type": 0,  # red wine
}

result = predict(sample_wine)
print(result)
# {'prediction': 0, 'quality_label': 'Low Quality'}
```

## Requirements

- Python 3.11+
- pandas >= 2.0
- scikit-learn >= 1.3
- optuna >= 3.5
- matplotlib >= 3.7
- joblib >= 1.3
- psutil >= 5.9
- mlflow >= 2.10 (optional)
- pytest >= 8.0

## Author

Student: [Your Name]

## License

This project is for educational purposes as part of a course assignment.
