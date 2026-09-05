import pickle
import pandas as pd
from prodml.config.config import get_settings

settings = get_settings()


class DurationPredictor:
    def __init__(
        self, model_bundle_path: str = settings.linear_regression_model_path
    ) -> None:
        self.model_bundle_path = model_bundle_path

    def load(self):
        with open(self.model_bundle_path, "rb") as f:
            feature_pipeline, model = pickle.load(f)

        return feature_pipeline, model

    def predict_one(self, raw_input_df: pd.DataFrame, feature_pipeline, model) -> list:
        """Applies exact feature transformation matrix adjustments and returns predictions."""
        X = feature_pipeline.transform(raw_input_df)
        return model.predict(X)

    def predict_batch(self):
        pass


if __name__ == "__main__":
    sample_payload = pd.DataFrame(
        [{"PULocationID": 198, "DOLocationID": 56, "trip_distance": 4.08}]
    )

    # pipeline = DurationPredictor("lr_baseline.pkl")
    # predictions = pipeline.predict_one(sample_payload)
    # print(f"Predicted Duration: {predictions[0]:.2f} minutes")
