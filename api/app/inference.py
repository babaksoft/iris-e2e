from pathlib import Path
import os

from joblib import load

from iris import IrisModel

feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_names = ["setosa", "versicolor", "virginica"]


def predict_iris(iris: IrisModel) -> str:
    api_root = Path(__file__).parent.parent
    model_path = api_root / "model/model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file could not be found.")

    model = load(model_path)
    scaler = model["scaler"]
    clf = model["classifier"]

    x = [float(iris.__dict__[col]) for col in feature_names]
    x_scaled = scaler.transform([x])
    prediction = clf.predict(x_scaled)
    return iris_names[prediction[0]]
