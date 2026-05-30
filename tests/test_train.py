import pandas as pd
import pytest

from src.etl import prepare_features


def test_train_pipeline_runs():
    """Test that the training pipeline can be initialized."""
    # Create a small synthetic dataset
    df = pd.DataFrame(
        {
            "fixed acidity": [7.4, 7.8, 7.0, 6.5, 7.2, 7.5],
            "volatile acidity": [0.70, 0.88, 0.50, 0.60, 0.65, 0.72],
            "citric acid": [0.00, 0.10, 0.30, 0.20, 0.25, 0.05],
            "residual sugar": [1.9, 2.5, 1.5, 1.0, 1.2, 2.0],
            "chlorides": [0.076, 0.080, 0.065, 0.070, 0.068, 0.078],
            "free sulfur dioxide": [11.0, 15.0, 10.0, 8.0, 9.0, 12.0],
            "total sulfur dioxide": [34.0, 40.0, 30.0, 25.0, 28.0, 35.0],
            "density": [0.9978, 0.9980, 0.9965, 0.9960, 0.9962, 0.9979],
            "pH": [3.51, 3.40, 3.30, 3.25, 3.28, 3.50],
            "sulphates": [0.56, 0.60, 0.50, 0.55, 0.53, 0.58],
            "alcohol": [9.4, 9.8, 11.0, 12.0, 11.5, 9.5],
            "wine_type": [0, 0, 1, 1, 1, 0],
            "quality": [5, 6, 8, 7, 7, 5],
        }
    )

    X, y = prepare_features(df)

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert set(y.unique()) == {0, 1}
