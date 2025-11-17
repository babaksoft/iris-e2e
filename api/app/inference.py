from pathlib import Path
import os

from joblib import load
import pandas as pd
from sklearn.datasets import load_iris

from iris import IrisModel


def get_iris_name(predicted: int):
    iris = load_iris()
    return iris.target_names[predicted]


def predict_iris(iris: IrisModel) -> str:
    api_root = Path(__file__).parent.parent
    model_path = api_root / "model/model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file could not be found.")

    model = load(model_path)
    scaler = model["scaler"]
    clf = model["classifier"]

    features = scaler.feature_names_in_
    x = [float(iris.__dict__[col]) for col in features]
    df = pd.DataFrame([x], columns=features)
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=features)
    prediction = clf.predict(df_scaled)
    return get_iris_name(prediction[0])
