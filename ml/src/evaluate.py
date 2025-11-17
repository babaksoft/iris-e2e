from pathlib import Path

from iris import IrisClassifier

def main():
    ml_root = Path(__file__).parent.parent
    data_ = ml_root / "data/prepared/test.csv"
    model_ = ml_root / "model/model.joblib"
    metrics_ = ml_root / "metrics/metrics.json"

    clf = IrisClassifier()
    clf.evaluate(data_, model_, metrics_)


if __name__ == "__main__":
    main()
