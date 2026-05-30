from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR


def predict(
    wine_properties: dict | pd.DataFrame,
    model_path: str | None = None,
    return_proba: bool = False,
) -> pd.DataFrame | list:
    """Predict wine quality for given wine properties.

    Args:
        wine_properties: Either a dict with wine properties or a DataFrame.
        model_path: Path to the trained model. Defaults to best_model.joblib.
        return_proba: If True, return prediction probabilities instead of classes.

    Returns:
        Predictions (0 = Low Quality, 1 = High Quality) or probabilities.
    """
    model_path = Path(model_path) if model_path else MODELS_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    import joblib

    model = joblib.load(model_path)

    # If single dict, convert to DataFrame
    if isinstance(wine_properties, dict):
        input_df = pd.DataFrame([wine_properties])
        single_input = True
    else:
        input_df = wine_properties.copy()
        single_input = False

    # Ensure correct column order
    expected_features = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "wine_type",
    ]

    available_features = [f for f in expected_features if f in input_df.columns]
    input_df = input_df[available_features]

    if return_proba:
        predictions = model.predict_proba(input_df)[:, 1]
        result = pd.DataFrame({
            "high_quality_probability": predictions
        })
    else:
        predictions = model.predict(input_df)
        result = pd.DataFrame({
            "prediction": predictions,
            "quality_label": ["High Quality" if p == 1 else "Low Quality" for p in predictions]
        })

    if single_input and len(result) == 1:
        return result.iloc[0].to_dict()

    return result


def predict_from_csv(
    csv_path: str,
    model_path: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Predict wine quality from a CSV file.

    Args:
        csv_path: Path to input CSV file.
        model_path: Path to the trained model.
        output_path: Path to save predictions CSV. If None, prints to console.

    Returns:
        DataFrame with predictions.
    """
    df = pd.read_csv(csv_path)
    predictions = predict(df, model_path=model_path)

    if output_path:
        output_df = df.copy()
        if isinstance(predictions, pd.DataFrame):
            output_df = pd.concat([output_df, predictions], axis=1)
        output_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

    return predictions


if __name__ == "__main__":
    # Example usage with sample wine properties
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
        "wine_type": 0,  # 0 = red, 1 = white
    }

    result = predict(sample_wine, return_proba=True)
    print(f"Prediction result: {result}")
