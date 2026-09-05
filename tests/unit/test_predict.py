import numpy as np
from prodml.predict import DurationPredictor


def test_predict_one_is_float_sane_and_deterministic(
    trained_model, sample_features, monkeypatch
):
    feature_pipeline, model = trained_model
    monkeypatch.setattr(
        DurationPredictor, "load", lambda self: (feature_pipeline, model)
    )

    predictor = DurationPredictor()
    first = predictor.predict_one(sample_features, feature_pipeline, model)
    second = predictor.predict_one(sample_features, feature_pipeline, model)

    assert isinstance(first[0], (float, np.floating))
    assert -30 <= first[0] <= 150
    assert np.allclose(first, second)
