from pathlib import Path

from iris import IrisClassifier


def main():
    ml_root = Path(__file__).parent.parent
    raw_ = ml_root / "data/raw/iris.csv"
    data_ = ml_root / "data/prepared/train.csv"
    model_ = ml_root / "model/model.joblib"

    clf = IrisClassifier()
    clf.prepare(raw_, model_)
    clf.train(data_, model_)

if __name__ == "__main__":
    main()
