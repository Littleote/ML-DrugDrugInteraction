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

# First nonstop word
FIRST_NONSTOPWORD_WORD = "first nonstop word (word)"
FIRST_NONSTOPWORD_LEMMA = "first nonstop word (lemma)"
FIRST_NONSTOPWORD_POSTAG = "first nonstop word (POS tag)"

# Words in between entities
NONSTOPWORDS_INBETWEEN_WORD = "all nonstop words in between (word)"
NONSTOPWORDS_INBETWEEN_LEMMA = "all nonstop words in between (lemma)"
BOOL_ENTITY_INBETWEEN = "entities in between (boolean)"
NUM_ENTITY_INBETWEEN = "entities in between (number)"
NUM_TOKEN_INBETWEEN = "number of words in between"
NUM_POSTAG_INBETWEEN = "count of each POS tag in between"

# Common ancestors
LOWEST_COMMON_SUBSUMER_REL = "lowest common subsumer (relation)"
LOWEST_COMMON_SUBSUMER_WORD = "lowest common subsumer (word)"
LOWEST_COMMON_SUBSUMER_LEMMA = "lowest common subsumer (lemma)"
LOWEST_COMMON_SUBSUMER_POSTAG = "lowest common subsumer (POS tag)"
PARENT_SPLIT_REL = "all common ancestors (relation)"
PARENT_SPLIT_WORD = "all common ancestors (word)"
PARENT_SPLIT_LEMMA = "all common ancestors (lemma)"
PARENT_SPLIT_LR = "all common ancestors (lemma + relation)"
NUM_TOKEN_PARENT = "number of common ancestors"
BOOL_COMMON_VERB = "lowest common verb (boolean)"
NUM_COMMON_VERB = "lowest common verb (distance)"
COMMON_VERB_WORD = "lowest common verb (word)"
COMMON_VERB_LEMMA = "lowest common verb (lemma)"
COMMON_VERB_REL = "lowest common verb (relation)"
COMMON_VERB_POSTAG = "lowest common verb (POS tag)"

# Drug-drug path
PATH_SPLIT_WORD = "drug-drug path words (word)"
PATH_SPLIT_LEMMA = "drug-drug path words (lemma)"
PATH_SPLIT_REL = "drug-drug path words (relation)"
PATH_SPLIT_LR = "drug-drug path words (lemma + relation)"
NUM_TOKEN_PATH = "number of words in drug-drug path"
NUM_POSTAG_PATH = "count of each POS tag in drug-drug path"
PATH_JOINT = "joined drug-drug path"

# Other
SAME_TOKEN = "are both entities in the same spot"

FEAT = {
    FIRST_NONSTOPWORD_WORD: False,
    FIRST_NONSTOPWORD_LEMMA: True,
    FIRST_NONSTOPWORD_POSTAG: False,
    NONSTOPWORDS_INBETWEEN_WORD: False,
    NONSTOPWORDS_INBETWEEN_LEMMA: True,
    BOOL_ENTITY_INBETWEEN: False,
    NUM_ENTITY_INBETWEEN: False,
    NUM_POSTAG_INBETWEEN: False,
    NUM_TOKEN_INBETWEEN: False,
    SAME_TOKEN: False,
    LOWEST_COMMON_SUBSUMER_REL: False,
    LOWEST_COMMON_SUBSUMER_WORD: False,
    LOWEST_COMMON_SUBSUMER_LEMMA: False,
    LOWEST_COMMON_SUBSUMER_POSTAG: False,
    PARENT_SPLIT_REL: False,
    PARENT_SPLIT_WORD: False,
    PARENT_SPLIT_LEMMA: True,
    PARENT_SPLIT_LR: False,
    NUM_TOKEN_PARENT: False,
    BOOL_COMMON_VERB: False,
    NUM_COMMON_VERB: False,
    COMMON_VERB_WORD: False,
    COMMON_VERB_LEMMA: False,
    COMMON_VERB_REL: False,
    COMMON_VERB_POSTAG: False,
    PATH_SPLIT_WORD: False,
    PATH_SPLIT_LEMMA: False,
    PATH_SPLIT_REL: False,
    PATH_SPLIT_LR: False,
    NUM_TOKEN_PATH: False,
    NUM_POSTAG_PATH: False,
    PATH_JOINT: True,
}

INBETWEEN = (
    False
    or FEAT[NONSTOPWORDS_INBETWEEN_WORD]
    or FEAT[NONSTOPWORDS_INBETWEEN_LEMMA]
    or FEAT[BOOL_ENTITY_INBETWEEN]
    or FEAT[NUM_ENTITY_INBETWEEN]
    or FEAT[NUM_POSTAG_INBETWEEN]
)

PARENT = (
    False
    or FEAT[PARENT_SPLIT_REL]
    or FEAT[PARENT_SPLIT_WORD]
    or FEAT[PARENT_SPLIT_LEMMA]
    or FEAT[PARENT_SPLIT_LR]
    or FEAT[NUM_TOKEN_PARENT]
    or FEAT[BOOL_COMMON_VERB]
    or FEAT[NUM_COMMON_VERB]
    or FEAT[COMMON_VERB_WORD]
    or FEAT[COMMON_VERB_LEMMA]
    or FEAT[COMMON_VERB_REL]
    or FEAT[COMMON_VERB_POSTAG]
)

PATH = (
    False
    or FEAT[PATH_SPLIT_WORD]
    or FEAT[PATH_SPLIT_LEMMA]
    or FEAT[PATH_SPLIT_REL]
    or FEAT[PATH_SPLIT_LR]
    or FEAT[NUM_TOKEN_PATH]
    or FEAT[NUM_POSTAG_PATH]
    or FEAT[PATH_JOINT]
)


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
            feats[FIRST_NONSTOPWORD_LEMMA].add(f"sw_lemma_{lemma}=1")
            feats[FIRST_NONSTOPWORD_WORD].add(f"sw_word_{word}=1")
            feats[FIRST_NONSTOPWORD_POSTAG].add(f"sw_POS_{tag}=1")
        except Exception:
            pass

        # feature indicating the presence of an entity in between E1 and E2
        #   ENTITY_INBETWEEN: True or False if there are entities in between
        #   ENTITIES_INBETWEEN: Total count of entities between the two tokens
        if INBETWEEN:
            counts = {}
            eib, num_eib = False, 0
            for tk in range(tkE1 + 1, tkE2):
                word = tree.get_word(tk)
                lemma = tree.get_lemma(tk).lower()
                tag = tree.get_tag(tk)
                counts[tag] = counts.get(tag, 0) + 1
                if not tree.is_stopword(tk):
                    feats[NONSTOPWORDS_INBETWEEN_LEMMA].add(f"swib_lemma_{lemma}=1")
                    feats[NONSTOPWORDS_INBETWEEN_WORD].add(f"swib_word_{word}=1")
                if tree.is_entity(tk, entities):
                    eib = True
                    num_eib += 1
            for tag, count in counts.items():
                feats[NUM_POSTAG_INBETWEEN].add(f"ptib_{tag}_num={count}")
            feats[BOOL_ENTITY_INBETWEEN].add(f"eib_{eib}=1")
            feats[NUM_ENTITY_INBETWEEN].add(f"eib_num={num_eib}")

        # feature indicating the presence of tokens in between E1 and E2
        #   TOKENS_INBETWEEN: Number of tokens that separate the two entities
        #   SAME_TOKEN: Both entities match to the same token
        feats[NUM_TOKEN_INBETWEEN].add(f"tib_num={max(0, tkE2 - tkE1 - 1)}")
        feats[SAME_TOKEN].add(f"tib_same={1 if tkE2 - tkE1 <= 0 else 0}")

        lcs = tree.get_LCS(tkE1, tkE2)
        lcs_lemma = tree.get_lemma(lcs).lower()
        lcs_rel = tree.get_rel(lcs).lower()
        lcs_word = tree.get_word(lcs).lower()
        lcs_tag = tree.get_tag(lcs)
        # features about the Lowest Common Subsumer
        #   LOWEST_COMMON_SUBSUMER: Information about the LCS (lowest common subsumer)
        feats[LOWEST_COMMON_SUBSUMER_LEMMA].add(f"lcs_lemma_{lcs_lemma}=1")
        feats[LOWEST_COMMON_SUBSUMER_REL].add(f"lcs_rel_{lcs_rel}=1")
        feats[LOWEST_COMMON_SUBSUMER_WORD].add(f"lcs_word_{lcs_word}=1")
        feats[LOWEST_COMMON_SUBSUMER_POSTAG].add(f"lcs_POS_{lcs_tag}=1")

        # features about common ancestors in the graph
        #   PARENT_SPLIT: Parent path (of LCS) element wise
        #   PARENT_LENGTH: Length of parent path (of LCS)
        #   COMMON_VERB: First (lowest) common verb among the two entities
        if PARENT:
            cv, cv_dist = lcs if lcs_tag.startswith("VB") else None, 0
            parents = tree.get_ancestors(lcs)
            feats[NUM_TOKEN_PARENT].add(f"parent_len={len(parents)}")
            for dist, x in enumerate(parents):
                word = tree.get_word(x).lower()
                lemma = tree.get_lemma(x).lower()
                rel = tree.get_rel(x).lower()
                tag = tree.get_tag(x)
                dist += 1
                feats[PARENT_SPLIT_WORD].add(f"parent_w_{word}=1")
                feats[PARENT_SPLIT_LEMMA].add(f"parent_l_{lemma}=1")
                feats[PARENT_SPLIT_LR].add(f"parent_lr_{lemma}_{rel}=1")
                feats[PARENT_SPLIT_REL].add(f"parent_r_{rel}=1")
                if cv is None and tag.startswith("VB"):
                    cv, cv_dist = x, dist + 1
            cv_lemma = tree.get_lemma(cv).lower()
            cv_rel = tree.get_rel(cv).lower()
            cv_word = tree.get_word(cv).lower()
            cv_tag = tree.get_tag(cv)
            feats[BOOL_COMMON_VERB].add(f"cv_present={0 if cv is None else 1}")
            if cv is not None:
                feats[NUM_COMMON_VERB].add(f"cv_dist={cv_dist}")
                feats[COMMON_VERB_LEMMA].add(f"cv_lemma_{cv_lemma}=1")
                feats[COMMON_VERB_REL].add(f"cv_rel_{cv_rel}=1")
                feats[COMMON_VERB_WORD].add(f"cv_word_{cv_word}=1")
                feats[COMMON_VERB_POSTAG].add(f"cv_POS_{cv_tag}=1")

        # features about paths in the tree
        #   PATH_JOINT: Full left, right and combined path
        #   PATH_SPLIT: Left and right path element wise
        #   PATH_LENGTH: Left and right path elements to entity
        if PATH:
            counts = {}
            path = tree.get_up_path(tkE1, lcs)
            path_steps = []
            for x in path:
                word = tree.get_word(x).lower()
                lemma = tree.get_lemma(x).lower()
                rel = tree.get_rel(x).lower()
                tag = tree.get_tag(x).lower()
                path_steps.append(f"{lemma}_{rel}")
                counts[tag] = counts.get(tag, 0) + 1
                feats[PATH_SPLIT_WORD].add(f"path1_w_{word}=1")
                feats[PATH_SPLIT_LEMMA].add(f"path1_l_{lemma}=1")
                feats[PATH_SPLIT_REL].add(f"path1_r_{rel}=1")
                feats[PATH_SPLIT_LR].add(f"path1_lr_{lemma}_{rel}=1")
            feats[NUM_TOKEN_PATH].add(f"path1_len={len(path)}")
            path1 = "<".join(path_steps)
            feats[PATH_JOINT].add(f"path1_{path1}=1")

            path = tree.get_down_path(lcs, tkE2)
            path_steps = []
            for x in path:
                word = tree.get_word(x).lower()
                lemma = tree.get_lemma(x).lower()
                rel = tree.get_rel(x).lower()
                tag = tree.get_tag(x).lower()
                path_steps.append(f"{lemma}_{rel}")
                counts[tag] = counts.get(tag, 0) + 1
                feats[PATH_SPLIT_WORD].add(f"path2_w_{word}=1")
                feats[PATH_SPLIT_LEMMA].add(f"path2_l_{lemma}=1")
                feats[PATH_SPLIT_REL].add(f"path2_r_{rel}=1")
                feats[PATH_SPLIT_LR].add(f"path2_lr_{lemma}_{rel}=1")
            feats[NUM_TOKEN_PATH].add(f"path2_len={len(path)}")
            path2 = ">".join(path_steps)
            feats[PATH_JOINT].add(f"path2_{path2}=1")

            path = f"{path1}<{lcs_lemma}_{lcs_rel}>{path2}"
            feats[PATH_JOINT].add(f"path_{path}=1")
            for tag, count in counts.items():
                feats[NUM_POSTAG_PATH].add(f"ptib_{tag}_num={count}")

    return (
        feats
        if split
        else set.union(*[feat for key, feat in feats.items() if FEAT[key]])
    )


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
