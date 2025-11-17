from pathlib import Path
import json

from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate(data_path, model_path, metrics_path: Path):
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


if __name__ == "__main__":
    ml_root = Path(__file__).parent.parent
    data_ = ml_root / "data/prepared/test.csv"
    model_ = ml_root / "model/model.joblib"
    metrics_ = ml_root / "metrics/metrics.json"
    evaluate(data_, model_, metrics_)
