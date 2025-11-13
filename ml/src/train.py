from pathlib import Path
from joblib import dump

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train(data_path, model_path):
    # Random state for repeatability
    rs = 147

    df_train = pd.read_csv(data_path)
    x_train = df_train.iloc[:, :-1]
    y_train = df_train[df_train.columns[-1]]

    # Start with a baseline model
    lr = LogisticRegression(solver="saga", random_state=rs)
    trained = lr.fit(x_train, y_train)
    dump(trained, model_path)


if __name__ == "__main__":
    ml_root = Path(__file__).parent.parent
    data_ = ml_root / "data/prepared/train.csv"
    model_ = ml_root / "model/model.joblib"
    train(data_, model_)
