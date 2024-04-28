#! /usr/bin/python3

import sys
from os import listdir
from typing import Any
from xml.dom.minidom import parse

import deptree as dtree

# import patterns


## -------------------
## -- Convert a pair of drugs and their context in a feature vector

FIRST_STOPWORD = False
ENTITY_INBETWEEN = False
ENTITIES_INBETWEEN = True
LOWEST_COMMON_SUBSUMER = True

PARENT_SPLIT = True
PARENT_LENGTH = True
COMMON_VERB = True
PARENT = PARENT_SPLIT or PARENT_LENGTH or COMMON_VERB

PATH_SPLIT = True
PATH_LENGTH = True
PATH_JOINT = False
PATH = PATH_SPLIT or PATH_LENGTH or PATH_JOINT


def extract_features(tree: dtree.deptree, entities: dict[Any, dict[str, int]], e1, e2):
    feats = set()

    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]["start"], entities[e1]["end"])
    tkE2 = tree.get_fragment_head(entities[e2]["start"], entities[e2]["end"])

    if tkE1 is not None and tkE2 is not None:
        if tkE1 > tkE2:
            tkE1, tkE2 = tkE2, tkE1

        if FIRST_STOPWORD:
            # features for tokens in between E1 and E2
            tk = tkE1 + 1
            try:
                while tree.is_stopword(tk):
                    tk += 1
                word = tree.get_word(tk)
                lemma = tree.get_lemma(tk).lower()
                tag = tree.get_tag(tk)
                feats.add(f"sw_lemma_{lemma}=1")
                feats.add(f"sw_word_{word}=1")
                feats.add(f"sw_POS_{tag}=1")
            except Exception:
                pass

        if ENTITY_INBETWEEN or ENTITIES_INBETWEEN:
            # feature indicating the presence of an entity in between E1 and E2
            eib, num_eib = False, 0
            for tk in range(tkE1 + 1, tkE2):
                if tree.is_entity(tk, entities):
                    eib = True
                    num_eib += 1
                    if not ENTITIES_INBETWEEN:
                        break
            if ENTITY_INBETWEEN:
                feats.add(f"eib_{eib}=1")
            if ENTITY_INBETWEEN:
                feats.add(f"eib_num={num_eib}")

        lcs = tree.get_LCS(tkE1, tkE2)
        lcs_lemma = tree.get_lemma(lcs).lower()
        lcs_rel = tree.get_rel(lcs).lower()
        lcs_word = tree.get_word(lcs).lower()
        lcs_tag = tree.get_tag(lcs)
        if LOWEST_COMMON_SUBSUMER:
            # features about the Lowest Common Subsumer
            feats.add(f"lcs_lemma_{lcs_lemma}=1")
            feats.add(f"lcs_rel_{lcs_rel}=1")
            feats.add(f"lcs_word_{lcs_word}=1")
            feats.add(f"lcs_POS_{lcs_tag}=1")

        # features about common ancestors in the graph
        cv, cv_dist = None, 0
        if PARENT:
            parent = tree.get_ancestors(lcs)
            if PARENT_LENGTH:
                feats.add(f"parent_len={len(parent)}")
            if PARENT_SPLIT or COMMON_VERB:
                for x in parent:
                    lemma = tree.get_lemma(x).lower()
                    rel = tree.get_rel(x).lower()
                    tag = tree.get_tag(x)
                    cv_dist += 1
                    if PARENT_SPLIT:
                        feats.add(f"parent_l_{lemma}=1")
                        feats.add(f"parent_lr_{lemma}_{rel}=1")
                    if COMMON_VERB and cv is None and tag.startswith("VB"):
                        cv = x
                        cv_lemma = tree.get_lemma(cv).lower()
                        cv_rel = tree.get_rel(cv).lower()
                        cv_word = tree.get_word(cv).lower()
                        cv_tag = tree.get_tag(cv)
                        feats.add("cv_present=1")
                        feats.add(f"cv_dist={cv_dist}")
                        feats.add(f"cv_lemma_{cv_lemma}=1")
                        feats.add(f"cv_rel_{cv_rel}=1")
                        feats.add(f"cv_word_{cv_word}=1")
                        feats.add(f"cv_POS_{cv_tag}=1")

        # features about paths in the tree
        if PATH:
            path = tree.get_up_path(tkE1, lcs)
            path_steps = []
            if PATH_JOINT or PATH_SPLIT:
                for x in path:
                    lemma = tree.get_lemma(x).lower()
                    rel = tree.get_rel(x).lower()
                    path_steps.append(f"{lemma}_{rel}")
                    if PATH_SPLIT:
                        feats.add(f"path1_l_{lemma}=1")
                        feats.add(f"path1_lr_{lemma}_{rel}=1")
            if PATH_LENGTH:
                feats.add(f"path1_len={len(path)}")
            if PATH_JOINT:
                path1 = "<".join(path_steps)
                feats.add(f"path1_{path1}=1")

            path = tree.get_down_path(lcs, tkE2)
            path_steps = []
            if PATH_JOINT or PATH_SPLIT:
                for x in path:
                    lemma = tree.get_lemma(x).lower()
                    rel = tree.get_rel(x).lower()
                    path_steps.append(f"{lemma}_{rel}")
                    if PATH_SPLIT:
                        feats.add(f"path2_l_{lemma}=1")
                        feats.add(f"path2_lr_{lemma}_{rel}=1")
            if PATH_LENGTH:
                feats.add(f"path2_len={len(path)}")
            if PATH_JOINT:
                path2 = ">".join(path_steps)
                feats.add(f"path2_{path2}=1")

            if PATH_JOINT:
                path = f"{path1}<{lcs_lemma}_{lcs_rel}>{path2}"
                feats.add(f"path_{path}=1")

    return feats


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]

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

            feats = extract_features(analysis, entities, id_e1, id_e2)
            # resulting vector
            if len(feats) != 0:
                print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")
