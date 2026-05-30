from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    HIGH_QUALITY_THRESHOLD,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load wine quality dataset from CSV."""
    data_path = path or __import__("src.config").config.RAW_DATA_PATH
    return pd.read_csv(data_path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the wine quality dataset."""
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for column in df.columns:
        if df[column].dtype in ["float64", "int64"]:
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else "unknown")

    return df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from existing columns.
    
    These features capture domain knowledge about wine chemistry:
    - Acid balance ratios
    - SO2 ratios
    - Density-acid interaction
    - Alcohol-acid interaction
    
    Only creates features for columns that exist in the DataFrame.
    """
    df = df.copy()
    
    # Acid balance: ratio of volatile to fixed acidity
    # Lower ratio indicates better acid balance (associated with higher quality)
    if "volatile acidity" in df.columns and "fixed acidity" in df.columns:
        df["acid_ratio"] = df["volatile acidity"] / (df["fixed acidity"] + 1e-8)
    
    # SO2 balance: ratio of free to total SO2
    # Higher ratio indicates fresher wine (associated with better quality preservation)
    if "free sulfur dioxide" in df.columns and "total sulfur dioxide" in df.columns:
        df["so2_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-8)
        df["free_so2_level"] = df["free sulfur dioxide"]
        df["total_so2_level"] = df["total sulfur dioxide"]
    
    # Chloride to density ratio (salinity indicator)
    if "chlorides" in df.columns and "density" in df.columns:
        df["salinity"] = df["chlorides"] / (df["density"] + 1e-8)
    
    # Alcohol to acid ratio (body indicator)
    # Higher alcohol with lower acid often indicates riper, better quality grapes
    if "alcohol" in df.columns and "fixed acidity" in df.columns:
        df["alcohol_acid_ratio"] = df["alcohol"] / (df["fixed acidity"] + 1e-8)
    
    # Citric acid presence (freshness indicator)
    if "citric acid" in df.columns:
        df["citric_presence"] = (df["citric acid"] > 0).astype(int)
    
    # Sulphates level (high sulphates often in better quality wines)
    if "sulphates" in df.columns:
        df["high_sulphates"] = (df["sulphates"] > df["sulphates"].median()).astype(int)
    
    # Residual sugar level (sweetness indicator) - use median split
    if "residual sugar" in df.columns:
        sugar_median = df["residual sugar"].median()
        df["high_sugar"] = (df["residual sugar"] > sugar_median).astype(int)
    
    # pH level category - use median split to avoid NaN
    if "pH" in df.columns:
        ph_median = df["pH"].median()
        df["high_ph"] = (df["pH"] > ph_median).astype(int)
    
    # Alcohol level (high alcohol often indicates better quality)
    if "alcohol" in df.columns:
        alcohol_median = df["alcohol"].median()
        df["high_alcohol"] = (df["alcohol"] > alcohol_median).astype(int)
    
    # Fixed acidity level
    if "fixed acidity" in df.columns:
        fixed_acid_median = df["fixed acidity"].median()
        df["high_fixed_acid"] = (df["fixed acidity"] > fixed_acid_median).astype(int)
    
    return df


def prepare_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    high_quality_threshold: int = HIGH_QUALITY_THRESHOLD,
):
    """Prepare features for classification task.

    Converts the regression target (quality score) to a binary classification:
    - 1: High quality wine (quality >= threshold)
    - 0: Low quality wine (quality < threshold)
    
    Creates engineered features that capture domain knowledge about wine chemistry.
    """
    df = clean_data(df)
    
    # Create engineered features
    df = create_engineered_features(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not found")

    # Create binary target
    y = (df[target_column] >= high_quality_threshold).astype(int)

    # Drop target and any unused columns
    drop_columns = [target_column]
    X = df.drop(columns=drop_columns)

    # Ensure only expected features are used
    available_features = [f for f in FEATURE_COLUMNS if f in X.columns]
    if "wine_type" in X.columns:
        available_features.append("wine_type")
    
    # Add engineered features
    engineered_features = [
        "acid_ratio",
        "so2_ratio",
        "free_so2_level",
        "total_so2_level",
        "salinity",
        "alcohol_acid_ratio",
        "citric_presence",
        "high_sulphates",
        "sugar_level",
        "ph_level",
    ]
    available_features.extend([f for f in engineered_features if f in X.columns])

    X = X[available_features]
    
    # Impute NaN values - engineered features like sugar_level and ph_level
    # may have NaN for outliers from pd.cut
    for column in X.columns:
        if X[column].dtype in ["float64", "int64", "float32", "int32"]:
            X[column] = X[column].fillna(X[column].median())

    return X, y.astype(int)


def split_dataset(X: pd.DataFrame, y: pd.Series):
    """Split dataset into train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_feature_groups(X: pd.DataFrame):
    """Identify numeric and categorical feature groups."""
    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    return numeric_features, categorical_features
