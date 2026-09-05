from pydantic_settings import BaseSettings
import os
from path import Path


class Settings(BaseSettings):
    app_name: str = "mlops"
    app_version: str = "0.0.1v"
    random_forest_model_path = (
        Path(__file__).parent.parent.parent.parent / "models" / "rf_baseline.pkl"
    )
    linear_regression_model_path = (
        Path(__file__).parent.parent.parent.parent / "models" / "lr_baseline.pkl"
    )
    dataset_path = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "green_tripdata_2026-05.parquet"
    )
    onnx_model_path = (
        Path(__file__).parent.parent.parent.parent / "models" / "lr_baseline.onnx"
    )

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        env_file_encoding = "utf-8"


def get_settings():
    return Settings()
