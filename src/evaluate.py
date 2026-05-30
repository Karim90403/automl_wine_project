from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MODELS_DIR, REPORTS_DIR, TABLES_DIR


def evaluate_model(model_path: str | None = None) -> dict:
    """Evaluate the trained model and generate evaluation report."""
    model_path = Path(model_path) if model_path else MODELS_DIR / "best_model.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    import joblib
    
    model = joblib.load(model_path)
    
    # Load test data - need to regenerate from config
    from src.download_data import ensure_dataset
    from src.etl import load_data, prepare_features
    from src.config import RAW_DATA_PATH, TEST_SIZE, RANDOM_STATE
    from sklearn.model_selection import train_test_split
    
    ensure_dataset()
    df = load_data()
    X, y = prepare_features(df)
    
    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    
    # Generate classification report
    classification_report_str = classification_report(y_test, y_pred)
    
    # Save evaluation report
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = TABLES_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Wine Quality Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report_str)
    
    print(f"Evaluation report saved to: {report_path}")
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
    
    return metrics


if __name__ == "__main__":
    evaluate_model()
