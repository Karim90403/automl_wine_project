import pandas as pd

from src.etl import clean_data, get_feature_groups, prepare_features, split_dataset


def test_clean_data_removes_duplicates():
    df = pd.DataFrame(
        {
            "fixed acidity": [7.4, 7.4, 7.8],
            "volatile acidity": [0.70, 0.70, 0.88],
            "quality": [5, 5, 6],
        }
    )

    cleaned = clean_data(df)
    assert len(cleaned) == 2


def test_clean_data_handles_missing_values():
    df = pd.DataFrame(
        {
            "fixed acidity": [7.4, None, 7.8],
            "volatile acidity": [0.70, 0.88, None],
            "quality": [5, 6, 7],
        }
    )

    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0


def test_prepare_features_creates_binary_target():
    df = pd.DataFrame(
        {
            "fixed acidity": [7.4, 7.8, 7.0, 6.5],
            "volatile acidity": [0.70, 0.88, 0.50, 0.60],
            "citric acid": [0.00, 0.10, 0.30, 0.20],
            "residual sugar": [1.9, 2.5, 1.5, 1.0],
            "chlorides": [0.076, 0.080, 0.065, 0.070],
            "free sulfur dioxide": [11.0, 15.0, 10.0, 8.0],
            "total sulfur dioxide": [34.0, 40.0, 30.0, 25.0],
            "density": [0.9978, 0.9980, 0.9965, 0.9960],
            "pH": [3.51, 3.40, 3.30, 3.25],
            "sulphates": [0.56, 0.60, 0.50, 0.55],
            "alcohol": [9.4, 9.8, 11.0, 12.0],
            "wine_type": [0, 0, 1, 1],
            "quality": [5, 6, 8, 7],
        }
    )

    X, y = prepare_features(df)

    assert "quality" not in X.columns
    assert set(y.tolist()) == {0, 1}


def test_split_dataset_keeps_rows():
    df = pd.DataFrame(
        {
            "fixed acidity": [7.0 + i * 0.1 for i in range(10)],
            "volatile acidity": [0.5 + i * 0.01 for i in range(10)],
            "quality": [5, 6, 7, 5, 6, 7, 8, 5, 6, 7],
            "wine_type": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_feature_groups_detect_numeric_and_categorical():
    X = pd.DataFrame({
        "fixed acidity": [7.4, 7.8],
        "alcohol": [9.4, 11.0],
        "wine_type": [0, 1],
    })
    numeric, categorical = get_feature_groups(X)

    assert "fixed acidity" in numeric
    assert "alcohol" in numeric
    assert "wine_type" in numeric  # wine_type is integer, not object/string
    assert len(categorical) == 0
