#!/usr/bin/env python3

import sys
from typing import Iterable
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# import argparse
from joblib import dump


def load_data(data: Iterable[str], *, start: int = 1, label: int | None = 0):
    features: list[dict[str, float]] = []
    get_label = label is not None
    labels: list[str] = []
    for interaction in data:
        interaction = interaction.strip(" \n\r\t")
        interaction = interaction.split("\t")
        interaction_dict = (
            {
                feat.split("=")[0]: float(feat.split("=")[1])
                for feat in interaction[start:]
            }
            if interaction[0] != ""
            else dict()
        )
        features.append(interaction_dict)
        if get_label:
            labels.append(interaction[label])
    return features, labels


def load_features(
    vectorizer_file: str,
    stdin: Iterable[str],
    feature_file: str | None = None,
    features: list[str] = None,
):
    split_features = feature_file is not None
    v = DictVectorizer()
    if split_features:
        X_train = {}
        _, y_train = load_data(stdin, start=4, label=3)
        for feat_name in features:
            file = feature_file.format(feat_name)
            with open(file, mode="r", encoding="utf8") as handler:
                feat, _ = load_data(handler, start=0, label=None)
                X_train[feat_name] = v.fit_transform(feat)
                dump(v, vectorizer_file.format(feat_name))
    else:
        train_features, y_train = load_data(stdin)
        X_train = v.fit_transform(train_features)
        dump(v, vectorizer_file)

    y_train = np.asarray(y_train)
    classes = np.unique(y_train)

    return (X_train, y_train), classes


def train_model(
    model_file: str,
    Xy: tuple[np.ndarray, np.ndarray],
    classes: np.ndarray,
):
    clf = MultinomialNB(alpha=0.01)
    clf.partial_fit(*Xy, classes)

    # Save Model
    dump(clf, model_file)


if __name__ == "__main__":
    model_file = sys.argv[1]
    vectorizer_file = sys.argv[2]
    Xy_train, classes = load_features(vectorizer_file, sys.stdin)
    train_model(model_file, Xy_train, classes)
