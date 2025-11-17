import json
import os

from joblib import load, dump
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class IrisClassifier:
    @staticmethod
    def ingest(to_path):
        cols = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "iris_type"
        ]
        iris = load_iris(as_frame=True)
        df = pd.concat([iris.data, iris.target], axis=1)
        df.columns = cols

        df.to_csv(to_path, header=True, index=False)

    @staticmethod
    def prepare(data_path, model_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "Raw data not found. Please call ingest method first."
            )

        # Set random state for repeatability
        rs = 147

        df = pd.read_csv(data_path)
        x = df.iloc[:, :-1]
        y = df[df.columns[-1]]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=rs
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        x_train_scaled = pd.DataFrame(
            x_train_scaled, columns=x_train.columns, index=x_train.index
        )
        x_test_scaled = pd.DataFrame(
            x_test_scaled, columns=x_test.columns, index=x_test.index
        )

        df_train = pd.concat([x_train_scaled, y_train], axis=1).round(6)
        df_test = pd.concat([x_test_scaled, y_test], axis=1).round(6)

        prepared = data_path.parent.parent / "prepared"
        df_train.to_csv(prepared / "train.csv", header=True, index=False)
        df_test.to_csv(prepared / "test.csv", header=True, index=False)

        model = {
            "scaler": scaler,
            "classifier": None
        }
        dump(model, model_path)

    @staticmethod
    def train(data_path, model_path):
        if not os.path.exists(data_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                "Required artifacts not found. "
                "Please call prepare method first."
            )

        # Random state for repeatability
        rs = 147

        df_train = pd.read_csv(data_path)
        x_train = df_train.iloc[:, :-1]
        y_train = df_train[df_train.columns[-1]]

        # Start with a baseline model
        lr = LogisticRegression(solver="saga", random_state=rs)
        trained = lr.fit(x_train, y_train)
        model = load(model_path)
        model["classifier"] = trained
        dump(model, model_path)

    @staticmethod
    def evaluate(data_path, model_path, metrics_path):
        if not os.path.exists(data_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                "Required artifacts not found. "
                "Please call train method first."
            )

        df_test = pd.read_csv(data_path)
        x_test = df_test.iloc[:, :-1]
        y_test = df_test[df_test.columns[-1]]
        model = load(model_path)["classifier"]

        y_predicted = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        f1 = f1_score(y_test, y_predicted, average="weighted")
        metrics = {
            "accuracy": round(accuracy, 4),
            "f1 score": round(f1, 4)
        }
        metrics_path.write_text(json.dumps(metrics))
