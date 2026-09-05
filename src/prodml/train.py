import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from prodml.config.config import get_settings
from prodml.data import Data
from prodml.features import FeaturePipeline

settings = get_settings()


class ModelTrainer:
    def run_training(self, data_path: str):
        data = Data(data_path=data_path)
        raw_df = data.data_load()

        pipeline = FeaturePipeline()
        processed_df = pipeline.compute_duration(raw_df)

        X, y = pipeline.fit_transform(processed_df)

        lr = LinearRegression()
        lr.fit(X, np.asarray(y))

        y_pred = lr.predict(X)
        self.save_model(settings.linear_regression_model_path, pipeline, lr)
        print(f"Train RMSE: {root_mean_squared_error(np.asarray(y), y_pred):.4f}")

        print("Exported pipeline and estimator state cleanly.")

    def save_model(self, model_name: str, pipeline: FeaturePipeline, model):
        with open(model_name, "wb") as f:
            pickle.dump((pipeline, model), f)


if __name__ == "__main__":
    settings = get_settings()
    model_trainer = ModelTrainer()
    model_trainer.run_training(settings.dataset_path)
