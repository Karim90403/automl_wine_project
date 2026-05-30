from __future__ import annotations

import time
import tracemalloc
from functools import wraps

import pandas as pd
from pathlib import Path

from src.config import TABLES_DIR


class ResourceMonitor:
    """Monitor resource usage during training."""

    def __init__(self):
        self.metrics = []

    def start(self):
        tracemalloc.start()
        self.start_time = time.time()
        import psutil
        self.process = psutil.Process()

    def stop(self) -> dict:
        tracemalloc.stop()
        end_time = time.time()
        import psutil

        current, peak = tracemalloc.get_traced_memory()
        return {
            "duration_seconds": end_time - self.start_time,
            "current_memory_mb": current / 1024 / 1024,
            "peak_memory_mb": peak / 1024 / 1024,
        }


def monitor_resources(func):
    """Decorator to monitor resource usage of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = ResourceMonitor()
        monitor.start()
        result = func(*args, **kwargs)
        metrics = monitor.stop()
        return result

    return wrapper


def save_resource_metrics(metrics: dict, filename: str = "resource_monitoring.csv") -> None:
    """Save resource monitoring metrics to CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(TABLES_DIR / filename, index=False)
