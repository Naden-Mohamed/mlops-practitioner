import pandas as pd


class Data:
    def __init__(self, data_path: str) -> None:
        self.dataset_path = data_path

    def data_load(self) -> pd.DataFrame:
        df = pd.read_parquet(self.dataset_path)
        return df

    def get_data_info(self, df: pd.DataFrame) -> dict:
        return {
            "dataset info": df.info(),
            "dataset size": df.size,
            "dataset columns": df.columns,
        }
