# features.py
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


class FeaturePipeline:
    def __init__(self, categorical: list | None = None, numerical: list | None = None):
        self.categorical = categorical or ["PULocationID", "DOLocationID"]
        self.numerical = numerical or ["trip_distance"]
        self.dv = DictVectorizer()

    def compute_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates target metric and handles clean filtering constraints."""
        df = df.copy()
        df["duration"] = (
            df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        ).dt.total_seconds() / 60.0
        return df[(df["duration"] >= 1) & (df["duration"] <= 120)]

    def fit_transform(self, df: pd.DataFrame):
        """Fits vectorizer and transforms raw text/integers to training matrices."""
        df = df.copy()
        df[self.categorical] = df[self.categorical].astype(str)
        dicts = df[self.categorical + self.numerical].to_dict(orient="records")

        X = self.dv.fit_transform(dicts)
        y = df["duration"].values if "duration" in df.columns else None
        return X, y

    def transform(self, df: pd.DataFrame):
        """Applies already-fitted transformer parameters to fresh real-time inference data."""
        df = df.copy()
        df[self.categorical] = df[self.categorical].astype(str)
        dicts = df[self.categorical + self.numerical].to_dict(orient="records")

        X = self.dv.transform(dicts)
        return X
