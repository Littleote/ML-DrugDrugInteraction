#!/usr/bin/env python3

import sys
from typing import Iterable
from joblib import load
# from sklearn.feature_extraction import DictVectorizer


def prepare_instances(xseq: Iterable[str]):
    features: list[dict[str, float]] = []
    for interaction in xseq:
        token_dict = {
            feat.split("=")[0]: float(feat.split("=")[1]) for feat in interaction[1:]
        }
        features.append(token_dict)
    return features


if __name__ == "__main__":
    # load leaned model and DictVectorizer
    model = load(sys.argv[1])
    v = load(sys.argv[2])

    for line in sys.stdin:
        fields = line.strip("\n").split("\t")
        (sid, e1, e2) = fields[0:3]
        vectors = v.transform(prepare_instances([fields[4:]]))
        prediction = model.predict(vectors)

        if prediction != "null":
            print(sid, e1, e2, prediction[0], sep="|")
