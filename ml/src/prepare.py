from pathlib import Path

from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare(path):
    # Set random state for repeatability
    rs = 147

    df = pd.read_csv(path)
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

    prepared = path.parent.parent / "prepared"
    df_train.to_csv(prepared / "train.csv", header=True, index=False)
    df_test.to_csv(prepared / "test.csv", header=True, index=False)

    model_path = path.parent.parent.parent / "model/model.joblib"
    model = {
        "scaler": scaler,
        "classifier": None
    }
    dump(model, model_path)


if __name__ == "__main__":
    ml_root = Path(__file__).parent.parent
    raw_data_path = ml_root / "data/raw/iris.csv"
    prepare(raw_data_path)
