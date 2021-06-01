import numpy as np
import pandas as pd
import requests
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV

# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer
# from sklearn.metrics import accuracy_score


DATA_DIR = "data"
DATA_FILENAME = "titanic.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/alexisperrier/packt-aml/master/ch4/titanic.csv"
)
PICKLE_FILENAME = "model.pkl"
TRANSFORMER_FILENAME = "transformer.pkl"


def download_data(target_path, url):
    r = requests.get(url, allow_redirects=True)
    return open(target_path, "wb").write(r.content)


def read_data(path):
    df = pd.read_csv(path, header=0, usecols=["sex", "age", "pclass", "survived"],)
    return df


def clean_data(df_in, dropna=False):
    if dropna:
        df_train = df_in.dropna().copy()
    else:
        df_train = df_in.copy()
    df_train["age"] = df_train.groupby(["pclass", "sex"]).age.transform(
        lambda x: x.fillna(x.mean())
    )
    df_train["age"] = df_train.age.fillna(value=df_train.age.mean())
    return df_train


def transform_features(df, transformer=None):
    columns = ["pclass", "sex", "age"]
    if transformer is None:
        model_num_transformers = Pipeline(
            [
                ("polynomial_features", PolynomialFeatures(2)),
                ("normalize", StandardScaler()),
            ]
        )
        transformer = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(dtype="int", drop="if_binary"), ["sex"]),
                ("numerical", model_num_transformers, ["age", "pclass"]),
            ],
            remainder="drop",
        )
        transformer.fit(df[columns])
    X_train = transformer.transform(df[columns])
    return transformer, X_train


def prepare_fit_data(df):
    df_features = df.drop(columns=["survived"])
    transformer, X_train = transform_features(df_features, transformer=None)
    y_train = df.survived
    return transformer, (X_train, y_train)


def fit_lr(X_train, y_train):
    clf = LogisticRegressionCV(
        penalty="l1",
        scoring="accuracy",
        solver="liblinear",
        Cs=20,
        cv=5,
        max_iter=500,
        random_state=0,
    ).fit(X_train, y_train)
    print("training score: " + str(clf.score(X_train, y_train)))
    # predictions = clf.predict_proba(X_test)[:, 0] > threshold
    return clf


def predict_proba(sex, age, pclass, transformer, clf):
    df = pd.DataFrame.from_records([{"sex": sex, "age": age, "pclass": pclass,}])
    _, X = transform_features(df, transformer=transformer)
    return clf.predict_proba(X)[0, 1]


DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)
MODEL_PATH = os.path.join(DATA_DIR, PICKLE_FILENAME)
TRANSFORMER_PATH = os.path.join(DATA_DIR, TRANSFORMER_FILENAME)


def load_model():
    with open(MODEL_PATH, "rb") as fd:
        model = pickle.load(fd)
    with open(TRANSFORMER_PATH, "rb") as fd:
        transformer = pickle.load(fd)
    return model, transformer


def run():
    try:
        os.mkdir(DATA_DIR)
    except:
        pass
    download_data(DATA_PATH, DATA_URL)
    transformer, (X, y) = prepare_fit_data(clean_data(read_data(DATA_PATH)))
    clf = fit_lr(X, y)
    with open(MODEL_PATH, "wb") as fd:
        pickle.dump(clf, fd)
    with open(TRANSFORMER_PATH, "wb") as fd:
        pickle.dump(transformer, fd)


if __name__ == "__main__":
    run()
