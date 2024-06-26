import glob
import itertools
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import scipy as sp

import predict_sklearn as predict
import train_sklearn as train
from evaluator import evaluate


F1_COL = "F1-Score"
FEAT_COL = "Used features"


def run(X_train, y_train, classes, model_file, features):
    if isinstance(list(X_train.values())[0], sp.sparse._base._spbase):
        X_train = sp.sparse.hstack([X_train[feat] for feat in features])
    elif isinstance(list(X_train.values())[0], np.ndarray):
        X_train = np.hstack([X_train[feat] for feat in features])
    else:
        raise ValueError(f"{type(list(X_train.values())[0])}")
    train.train_model(model_file, (X_train, y_train), classes)


def test(X_test, ids, model_file, data_dir, output_file, stats_file, features):
    model = predict.get_model(model_file)
    if isinstance(list(X_test.values())[0], sp.sparse._base._spbase):
        X_test = sp.sparse.hstack([X_test[feat] for feat in features])
    elif isinstance(list(X_test.values())[0], np.ndarray):
        X_test = np.hstack([X_test[feat] for feat in features])
    else:
        raise ValueError(f"{type(list(X_test.values())[0])}")
    with open(output_file, mode="w", encoding="utf-8") as stdout:
        predict.predict(model, ids, X_test, stdout)
    with open(stats_file, mode="w", encoding="utf-8") as stdout:
        evaluate("DDI", data_dir, output_file, None, redirect=stdout, precision=10)
    with open(stats_file, mode="r", encoding="utf-8") as handler:
        f1_score = handler.readlines()[7].split("\t")[-1]
        f1_score = float(f1_score.strip(" \n\t\r%"))
    return f1_score


def get_subsets(complete: list[str]):
    if len(complete) > 1:
        for i in range(len(complete)):
            yield complete[:i] + complete[i + 1 :]


def main(
    train_template: str,
    train_info_file: str,
    test_template: str,
    test_info_file: str,
    test_data_dir: str,
    vect_template: str,
    model_file: str,
    output_file: str,
    stats_file: str,
    redirect: str | TextIO,
    features_names: list[str],
):
    with open(train_info_file, mode="r", encoding="utf-8") as train_labels:
        (X_train, y_train), classes = train.load_features(
            vect_template, train_labels, train_template, features_names
        )
    with open(test_info_file, mode="r", encoding="utf-8") as test_ids:
        ids, X_test = predict.load_vectors(
            vect_template, test_ids, test_template, features_names
        )
    used_features: list[str] = []
    steps: list[tuple] = []
    best_f1_score = 0
    last_change: str | None = None
    random.shuffle(features_names)
    for feature in itertools.cycle(features_names):
        if feature == last_change:
            break
        if feature in used_features:
            continue
        used_features.append(feature)
        run(X_train, y_train, classes, model_file, used_features)
        new_f1_score = test(
            X_test,
            ids,
            model_file,
            test_data_dir,
            output_file,
            stats_file,
            used_features,
        )
        if new_f1_score > best_f1_score:
            last_change = feature
            best_f1_score = new_f1_score
            steps.append((best_f1_score, tuple(used_features)))
            trim = "The best subset of features"
            while trim is not None:
                trim = None
                for subset in get_subsets(used_features):
                    run(X_train, y_train, classes, model_file, subset)
                    new_f1_score = test(
                        X_test,
                        ids,
                        model_file,
                        test_data_dir,
                        output_file,
                        stats_file,
                        subset,
                    )
                    if new_f1_score > best_f1_score:
                        best_f1_score = new_f1_score
                        trim = subset
                if trim is not None:
                    used_features = trim
                    steps.append((best_f1_score, tuple(used_features)))
        else:
            used_features.remove(feature)
    df = pd.DataFrame(steps, columns=[F1_COL, FEAT_COL])
    df[F1_COL] = df[F1_COL].apply(lambda x: f"{x:0<7.6}%")

    def unify_parenthesis(elems: list[str]):
        keys = {}
        for elem in elems:
            key = elem.split("(", 1)[0]
            group = keys.get(key, [])
            group.append(elem)
            keys[key] = group
        elems = []
        for key, group in keys.items():
            if len(group) > 1:
                joined = (
                    key
                    + "("
                    + ", ".join(map(lambda x: x.split("(", 1)[1].strip("()"), group))
                    + ")"
                )
                elems.append(joined)
            else:
                elems.append(group[0])
        return elems

    df[FEAT_COL] = df[FEAT_COL].apply(unify_parenthesis).apply(lambda x: ", ".join(x))
    df[FEAT_COL] = df[FEAT_COL]
    df.to_csv(redirect)


if __name__ == "__main__":
    assert len(sys.argv) > 5, f"Expected at least 5 arguments found {len(sys.argv) - 1}"
    train_file = sys.argv[1]
    train_ids = sys.argv[2]
    train_glob = train_file.replace("{}", "*")
    train_regex = (
        train_file.replace(
            "\\",
            "\\\\",
        )
        .replace(
            ".",
            "\\.",
        )
        .replace("{}", "(.*)")
    )
    features = [
        re.match(train_regex, str(file)).group(1) for file in glob.glob(train_glob)
    ]
    test_file = sys.argv[3]
    test_ids = sys.argv[4]
    test_dir = sys.argv[5]
    redirect = sys.argv[6] if len(sys.argv) > 6 else sys.stdout
    with tempfile.TemporaryDirectory() as tmp_dir:
        vect_file = str(Path(tmp_dir, "vectorizer_{}.joblib"))
        model_file = str(Path(tmp_dir, "model.joblib"))
        output_file = str(Path(tmp_dir, "devel.out"))
        stats_file = str(Path(tmp_dir, "devel.stats"))
        main(
            train_template=train_file,
            train_info_file=train_ids,
            test_template=test_file,
            test_info_file=test_ids,
            test_data_dir=test_dir,
            vect_template=vect_file,
            model_file=model_file,
            output_file=output_file,
            stats_file=stats_file,
            redirect=redirect,
            features_names=features,
        )
