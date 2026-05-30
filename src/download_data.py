from __future__ import annotations

import pandas as pd
import urllib.request
from pathlib import Path

from src.config import (
    COMBINED_DATA_NAME,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    RED_WINE_URL,
    WHITE_WINE_URL,
)


def download_wine_dataset(force: bool = False) -> Path:
    """Download red and white wine quality datasets and combine them."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and not force:
        print(f"Dataset already exists: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    # Download red wine data
    red_wine_path = RAW_DATA_DIR / "winequality-red.csv"
    print(f"Downloading red wine dataset from {RED_WINE_URL}")
    urllib.request.urlretrieve(RED_WINE_URL, red_wine_path)

    # Download white wine data
    white_wine_path = RAW_DATA_DIR / "winequality-white.csv"
    print(f"Downloading white wine dataset from {WHITE_WINE_URL}")
    urllib.request.urlretrieve(WHITE_WINE_URL, white_wine_path)

    # Load and combine datasets
    red_wine = pd.read_csv(red_wine_path, sep=";")
    red_wine["wine_type"] = 0  # 0 for red

    white_wine = pd.read_csv(white_wine_path, sep=";")
    white_wine["wine_type"] = 1  # 1 for white

    combined = pd.concat([red_wine, white_wine], ignore_index=True)

    # Save combined dataset
    combined.to_csv(RAW_DATA_PATH, index=False, sep=";")

    # Clean up individual files
    red_wine_path.unlink(missing_ok=True)
    white_wine_path.unlink(missing_ok=True)

    print(f"Combined dataset saved to {RAW_DATA_PATH}")
    print(f"Total rows: {len(combined)}")
    print(f"Columns: {list(combined.columns)}")

    return RAW_DATA_PATH


def ensure_dataset() -> Path:
    """Ensure dataset exists, download if necessary."""
    if not RAW_DATA_PATH.exists():
        return download_wine_dataset(force=False)
    return RAW_DATA_PATH


if __name__ == "__main__":
    download_wine_dataset(force=True)
