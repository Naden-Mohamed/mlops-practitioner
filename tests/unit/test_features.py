import numpy as np
import pytest


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"PULocationID": None}, id="missing_category"),
        pytest.param({"trip_distance": 0.0}, id="zero_distance"),
        pytest.param(
            {"PULocationID": 999999, "DOLocationID": 999999}, id="unseen_pu_do_pair"
        ),
    ],
)
def test_transform_handles_edge_cases(trained_model, sample_features, overrides):
    feature_pipeline, _ = trained_model
    row = sample_features.iloc[[0]].copy()
    for col, val in overrides.items():
        row[col] = val

    X = feature_pipeline.transform(row)
    assert X.shape == (1, len(feature_pipeline.dv.feature_names_))
    assert np.isfinite(X.toarray()).all()
