#! /usr/bin/python3

from contextlib import ExitStack
import sys
from os import listdir
from typing import Any
from xml.dom.minidom import parse

import deptree as dtree

# import patterns


## -------------------
## -- Convert a pair of drugs and their context in a feature vector

FIRST_STOPWORD = "FIRST_STOPWORD"
ENTITY_INBETWEEN = "ENTITY_INBETWEEN"
ENTITIES_INBETWEEN = "ENTITIES_INBETWEEN"
TOKENS_INBETWEEN = "TOKENS_INBETWEEN"
SAME_TOKEN = "SAME_TOKEN"
LOWEST_COMMON_SUBSUMER = "LOWEST_COMMON_SUBSUMER"

PARENT_SPLIT = "PARENT_SPLIT"
PARENT_LENGTH = "PARENT_LENGTH"
COMMON_VERB = "COMMON_VERB"

PATH_SPLIT = "PATH_SPLIT"
PATH_LENGTH = "PATH_LENGTH"
PATH_JOINT = "PATH_JOINT"

FEAT = {
    FIRST_STOPWORD: False,
    ENTITY_INBETWEEN: False,
    ENTITIES_INBETWEEN: True,
    TOKENS_INBETWEEN: True,
    SAME_TOKEN: True,
    LOWEST_COMMON_SUBSUMER: True,
    PARENT_SPLIT: True,
    PARENT_LENGTH: True,
    COMMON_VERB: True,
    PATH_SPLIT: True,
    PATH_LENGTH: True,
    PATH_JOINT: False,
}

ENTITIES = FEAT[ENTITY_INBETWEEN] or FEAT[ENTITIES_INBETWEEN]
PATH = FEAT[PATH_SPLIT] or FEAT[PATH_LENGTH] or FEAT[PATH_JOINT]
PARENT = FEAT[PARENT_SPLIT] or FEAT[PARENT_LENGTH] or FEAT[COMMON_VERB]


def extract_features(
    tree: dtree.deptree, entities: dict[Any, dict[str, int]], e1, e2, split=False
):
    feats = {key: set() for key in FEAT.keys()}

    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]["start"], entities[e1]["end"])
    tkE2 = tree.get_fragment_head(entities[e2]["start"], entities[e2]["end"])

    if tkE1 is not None and tkE2 is not None:
        if tkE1 > tkE2:
            tkE1, tkE2 = tkE2, tkE1

        # features for tokens in between E1 and E2
        #   FIRST_STOPWORD: First noun, verb, adjective or ¿R? after the first token
        tk = tkE1 + 1
        try:
            while tree.is_stopword(tk):
                tk += 1
            word = tree.get_word(tk)
            lemma = tree.get_lemma(tk).lower()
            tag = tree.get_tag(tk)
            feats[FIRST_STOPWORD].add(f"sw_lemma_{lemma}=1")
            feats[FIRST_STOPWORD].add(f"sw_word_{word}=1")
            feats[FIRST_STOPWORD].add(f"sw_POS_{tag}=1")
        except Exception:
            pass

        # feature indicating the presence of an entity in between E1 and E2
        #   ENTITY_INBETWEEN: True or False if there are entities in between
        #   ENTITIES_INBETWEEN: Total count of entities between the two tokens
        if ENTITIES:
            eib, num_eib = False, 0
            for tk in range(tkE1 + 1, tkE2):
                if tree.is_entity(tk, entities):
                    eib = True
                    num_eib += 1
            feats[ENTITY_INBETWEEN].add(f"eib_{eib}=1")
            feats[ENTITIES_INBETWEEN].add(f"eib_num={num_eib}")

        # feature indicating the presence of tokens in between E1 and E2
        #   TOKENS_INBETWEEN: Number of tokens that separate the two entities
        #   SAME_TOKEN: Both entities match to the same token
        feats[TOKENS_INBETWEEN].add(f"tib_num={max(0, tkE2 - tkE1 - 1)}")
        feats[SAME_TOKEN].add(f"tib_same={1 if tkE2 - tkE1 <= 0 else 0}")

        lcs = tree.get_LCS(tkE1, tkE2)
        lcs_lemma = tree.get_lemma(lcs).lower()
        lcs_rel = tree.get_rel(lcs).lower()
        lcs_word = tree.get_word(lcs).lower()
        lcs_tag = tree.get_tag(lcs)
        # features about the Lowest Common Subsumer
        #   LOWEST_COMMON_SUBSUMER: Information about the LCS (lowest common subsumer)
        feats[LOWEST_COMMON_SUBSUMER].add(f"lcs_lemma_{lcs_lemma}=1")
        feats[LOWEST_COMMON_SUBSUMER].add(f"lcs_rel_{lcs_rel}=1")
        feats[LOWEST_COMMON_SUBSUMER].add(f"lcs_word_{lcs_word}=1")
        feats[LOWEST_COMMON_SUBSUMER].add(f"lcs_POS_{lcs_tag}=1")

        # features about common ancestors in the graph
        #   PARENT_SPLIT: Parent path (of LCS) element wise
        #   PARENT_LENGTH: Length of parent path (of LCS)
        #   COMMON_VERB: First (lowest) common verb among the two entities
        if PARENT:
            cv, cv_dist = lcs if lcs_tag.startswith("VB") else None, 0
            parents = tree.get_ancestors(lcs)
            feats[PARENT_LENGTH].add(f"parent_len={len(parents)}")
            for dist, x in enumerate(parents):
                lemma = tree.get_lemma(x).lower()
                rel = tree.get_rel(x).lower()
                tag = tree.get_tag(x)
                dist += 1
                feats[PARENT_SPLIT].add(f"parent_l_{lemma}=1")
                feats[PARENT_SPLIT].add(f"parent_lr_{lemma}_{rel}=1")
                feats[PARENT_SPLIT].add(f"parent_r_{rel}=1")
                if cv is None and tag.startswith("VB"):
                    cv, cv_dist = x, dist + 1
            cv_lemma = tree.get_lemma(cv).lower()
            cv_rel = tree.get_rel(cv).lower()
            cv_word = tree.get_word(cv).lower()
            cv_tag = tree.get_tag(cv)
            feats[COMMON_VERB].add(f"cv_present={0 if cv is None else 1}")
            if cv is not None:
                feats[COMMON_VERB].add(f"cv_dist={cv_dist}")
                feats[COMMON_VERB].add(f"cv_lemma_{cv_lemma}=1")
                feats[COMMON_VERB].add(f"cv_rel_{cv_rel}=1")
                feats[COMMON_VERB].add(f"cv_word_{cv_word}=1")
                feats[COMMON_VERB].add(f"cv_POS_{cv_tag}=1")

        # features about paths in the tree
        #   PATH_JOINT: Full left, right and combined path
        #   PATH_SPLIT: Left and right path element wise
        #   PATH_LENGTH: Left and right path elements to entity
        if PATH:
            path = tree.get_up_path(tkE1, lcs)
            path_steps = []
            for x in path:
                lemma = tree.get_lemma(x).lower()
                rel = tree.get_rel(x).lower()
                path_steps.append(f"{lemma}_{rel}")
                feats[PATH_SPLIT].add(f"path1_l_{lemma}=1")
                feats[PATH_SPLIT].add(f"path1_lr_{lemma}_{rel}=1")
                feats[PATH_SPLIT].add(f"path1_r_{rel}=1")
            feats[PATH_LENGTH].add(f"path1_len={len(path)}")
            path1 = "<".join(path_steps)
            feats[PATH_JOINT].add(f"path1_{path1}=1")

            path = tree.get_down_path(lcs, tkE2)
            path_steps = []
            for x in path:
                lemma = tree.get_lemma(x).lower()
                rel = tree.get_rel(x).lower()
                path_steps.append(f"{lemma}_{rel}")
                feats[PATH_SPLIT].add(f"path2_l_{lemma}=1")
                feats[PATH_SPLIT].add(f"path2_lr_{lemma}_{rel}=1")
                feats[PATH_SPLIT].add(f"path2_r_{rel}=1")
            feats[PATH_LENGTH].add(f"path2_len={len(path)}")
            path2 = ">".join(path_steps)
            feats[PATH_JOINT].add(f"path2_{path2}=1")

            path = f"{path1}<{lcs_lemma}_{lcs_rel}>{path2}"
            feats[PATH_JOINT].add(f"path_{path}=1")

    return feats if split else set.union(*feats.values())


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]

# output feature folder
feature_file = sys.argv[2] if len(sys.argv) > 2 else None
split_features = feature_file is not None

with ExitStack() as stack:
    files = (
        {
            feature: stack.enter_context(
                open(
                    feature_file.format(feature),
                    mode="w",
                    encoding="utf-8",
                )
            )
            for feature in FEAT.keys()
        }
        if split_features
        else None
    )

    # process each file in directory
    for f in listdir(datadir):
        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)

        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            stext = s.attributes["text"].value  # get sentence text
            # load sentence entities
            entities: dict[Any, dict[str, int]] = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                id = e.attributes["id"].value
                offs = e.attributes["charOffset"].value.split("-")
                entities[id] = {"start": int(offs[0]), "end": int(offs[-1])}

            # there are no entity pairs, skip sentence
            if len(entities) <= 1:
                continue

            # analyze sentence
            analysis = dtree.deptree(stext)

            # for each pair in the sentence, decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                # ground truth
                ddi = p.attributes["ddi"].value
                if ddi == "true":
                    dditype = p.attributes["type"].value
                else:
                    dditype = "null"
                # target entities
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                # feature extraction

                feats = extract_features(
                    analysis,
                    entities,
                    id_e1,
                    id_e2,
                    split_features,
                )

                has_features = len(feats) > 0
                if split_features:
                    has_features = any(map(lambda x: len(x) > 0, feats.values()))
                    if has_features:
                        for key, feat in feats.items():
                            print("\t".join(feat), file=files[key])
                    feats = []

                # resulting vector
                if has_features:
                    print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")