#!/usr/bin/env python3

import sys
from typing import Iterable, TextIO
from joblib import load
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB


from train_sklearn import load_data


def prepare_instances(xseq: Iterable[list[str]]):
    features: list[dict[str, float]] = []
    for interaction in xseq:
        token_dict = {
            feat.split("=")[0]: float(feat.split("=")[1]) for feat in interaction
        }
        features.append(token_dict)
    return features


def load_vectors(
    vectorizer_file: str,
    stdin: Iterable[str],
    feature_file: str = None,
    features: list[str] = None,
):
    split_features = feature_file is not None
    details = list(map(lambda x: (x.split("\t")[:3], x.split("\t")[4:]), stdin))
    if split_features:
        vectors = {}
        for feat_name in features:
            f_file = feature_file.format(feat_name)
            v_file = vectorizer_file.format(feat_name)
            v: DictVectorizer = load(v_file)
            with open(f_file, mode="r", encoding="utf8") as handler:
                feat, _ = load_data(handler, start=0, label=None)
                vectors[feat_name] = v.transform(feat)
    else:
        feat = list(map(lambda x: x[1], details))
        v: DictVectorizer = load(vectorizer_file)
        vectors = v.transform(prepare_instances(feat))

    return list(map(lambda x: x[0], details)), vectors


def predict(
    model: MultinomialNB, ids: list[list[str]], vectors: np.ndarray, stdout: TextIO
):
    predictions = model.predict(vectors)
    for _id, pred in zip(ids, predictions):
        print(*_id, pred, sep="|", file=stdout)


def get_model(model_file: str) -> MultinomialNB:
    return load(model_file)


if __name__ == "__main__":
    # load leaned model and DictVectorizer
    model = get_model(sys.argv[1])
    ids, vectors = load_vectors(sys.argv[2], sys.stdin)
    predict(model, ids, vectors, sys.stdout)
