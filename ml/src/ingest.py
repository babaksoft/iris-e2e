from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def ingest(to_path):
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_type"]
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target], axis=1)
    df.columns = cols

    df.to_csv(to_path, header=True, index=False)


if __name__ == "__main__":
    ml_root = Path(__file__).parent.parent
    data_path = ml_root / "data/raw/iris.csv"
    ingest(data_path)
