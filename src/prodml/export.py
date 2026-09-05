from typing import cast

from onnx import ModelProto
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config.config import get_settings
from prodml.predict import DurationPredictor


def export_to_onnx(pickly_path: str, onnx_path: str):
    feature_pipeline, model = DurationPredictor(pickly_path).load()
    num_features = len(feature_pipeline.dv.feature_names_)
    initial_type = [("float_input", FloatTensorType([None, num_features]))]
    onnx_model = cast(
        ModelProto,
        convert_sklearn(model, initial_types=initial_type, target_opset=17),
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return onnx_model


if __name__ == "__main__":
    settings = get_settings()
    export_to_onnx(settings.linear_regression_model_path, settings.onnx_model_path)
    print("Exported lr_baseline.onnx")
