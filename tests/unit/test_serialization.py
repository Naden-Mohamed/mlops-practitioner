import numpy as np

from prodml.export import vectorize_for_onnx
from prodml.config.config import get_settings

N_VALIDATION_ROWS = 500
ATOL = 1e-4

settings = get_settings()


def test_onnx_predictions_match_pickle(validation_rows, bundle, onnx_session):
    feature_pipeline, model = bundle
    X_dense = vectorize_for_onnx(feature_pipeline, validation_rows)

    pred_pkl = model.predict(X_dense)
    input_name = onnx_session.get_inputs()[0].name
    pred_onnx = onnx_session.run(None, {input_name: X_dense})[0].reshape(-1)

    assert np.allclose(pred_pkl, pred_onnx, atol=ATOL)
