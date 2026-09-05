import pickle
import onnxruntime as rt
from fastapi.testclient import TestClient
import pytest
import pandas as pd

from prodml.data import Data
from prodml.export import export_to_onnx
from prodml.config.config import get_settings

N_VALIDATION_ROWS = 500
ATOL = 1e-4

settings = get_settings()


@pytest.fixture
def validation_rows():
    df = Data(data_path=settings.dataset_path).data_load()
    return df.sample(n=N_VALIDATION_ROWS, random_state=42)


@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"PULocationID": 1, "DOLocationID": 3, "trip_distance": 3.2},
            {"PULocationID": 2, "DOLocationID": 4, "trip_distance": 1.1},
        ]
    )


@pytest.fixture(scope="session")  # session-scoped means it's built once and reused
def model():
    with open(settings.linear_regression_model_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def onnx_session():
    export_to_onnx(settings.linear_regression_model_path, settings.onnx_model_path)
    return rt.InferenceSession(
        settings.onnx_model_path, providers=["CPUExecutionProvider"]
    )


@pytest.fixture
def client():
    from ..main import app

    return TestClient(app=app)
