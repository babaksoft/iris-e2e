from pathlib import Path

from iris import IrisClassifier


def main():
    ml_root = Path(__file__).parent.parent
    data_path = ml_root / "data/raw/iris.csv"

    clf = IrisClassifier()
    clf.ingest(data_path)


if __name__ == "__main__":
    main()
