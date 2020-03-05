#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Data Loader/Pre-processor Functions

Feel free to change/restructure the code below
"""
import codecs, json, os, string
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from collections import defaultdict


"""Useful constants when processing `relations.json`"""
ARG1 = 'Arg1'
ARG2 = 'Arg2'
CONN = 'Connective'
SENSE = 'Sense'
TYPE = 'Type'
KEYS = [ARG1, ARG2, CONN, SENSE]
TEXT = 'RawText'
label_ref = {}
MAX_LEN_1 = 75
MAX_LEN_2 = 50
MAX_LEN_CON = 10
vocab = {"": 0}
output_rels = []


def preprocess(rel):
    """Preprocesses a single relation in `relations.json`

  Args:
    rel: dict

  Returns:
    see `featurize` above
  """
    rel_dict = {}
    outputs = defaultdict(dict)
    for key in KEYS:
        if key in [ARG1, ARG2, CONN]:
            # for `Arg1`, `Arg2`, `Connective`, we only keep tokens of `RawText`
            rel_dict[key] = rel[key][TEXT]
            # get token list
            if len(np.array(rel[key]["TokenList"])) > 2:
                outputs[key]["TokenList"] = np.array(rel[key]["TokenList"])[:,2]
            else:
                outputs[key]["TokenList"] = np.array([])
        elif key == SENSE:
            # `Sense` is the target label. For relations with multiple senses, we
            # assume (for simplicity) that the first label is the gold standard.
            rel_dict[key] = rel[key][0]

    outputs["DocID"] = rel["DocID"]
    outputs[ARG1]["TokenList"] = outputs[ARG1]["TokenList"][0:MAX_LEN_1]
    outputs[ARG2]["TokenList"] = outputs[ARG2]["TokenList"][0:MAX_LEN_2]
    outputs[CONN]["TokenList"] = outputs[CONN]["TokenList"][0:MAX_LEN_CON]
    outputs["Type"] = rel["Type"]

    # integer encode the document
    arg1s = rel_dict[ARG1].lower().split()[0:MAX_LEN_1]
    arg2s = rel_dict[ARG2].lower().split()[0:MAX_LEN_2]
    connectives = rel_dict[CONN].lower().split()[0:MAX_LEN_CON]
    # normalization
    arg1s = normalization(arg1s)
    arg2s = normalization(arg2s)
    connectives = normalization(connectives)

    # padding
    arg1s += [""] * (MAX_LEN_1-len(arg1s))
    arg2s += [""] * (MAX_LEN_2-len(arg2s))
    connectives += [""] * (MAX_LEN_CON-len(connectives))
    outputs[ARG1]["TokenList"] = np.pad(outputs[ARG1]["TokenList"], (0, MAX_LEN_1-outputs[ARG1]["TokenList"].shape[0]), 'constant')
    outputs[ARG2]["TokenList"] = np.pad(outputs[ARG2]["TokenList"], (0, MAX_LEN_2 - outputs[ARG2]["TokenList"].shape[0]), 'constant')
    outputs[CONN]["TokenList"] = np.pad(outputs[CONN]["TokenList"], (0, MAX_LEN_CON - outputs[CONN]["TokenList"].shape[0]), 'constant')

    # add label to labels list
    if rel_dict[SENSE] not in label_ref:
        label_ref[rel_dict[SENSE]] = len(label_ref)

    tokens = arg1s + connectives + arg2s

    # tranfer lists in outputs from numpy array to list
    outputs[ARG1]["TokenList"] = outputs[ARG1]["TokenList"].tolist()
    outputs[ARG2]["TokenList"] = outputs[ARG2]["TokenList"].tolist()
    outputs[CONN]["TokenList"] = outputs[CONN]["TokenList"].tolist()
    return tokens, label_ref[rel_dict[SENSE]], outputs


def remove_stop_words(tokens: list) -> list:
    """
  remove stop words from the token list
  :param tokens: a list of tokens
  :return: a list of tokens with stopwords removed
  """
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]


def normalization(tokens: list) -> list:
    """
    remove any punctuations at the end
    :param tokens: a list of tokens
    :return: a list of normalized tokens
    """
    tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
    return tokens


def get_vocab(train_path):
    rel_path = os.path.join(train_path, "relations.json")
    assert os.path.exists(rel_path), \
        "{} does not exist in `load_relations.py".format(rel_path)

    with codecs.open(rel_path, encoding='utf-8') as pdtb:
        for pdtb_line in pdtb:
            rel = json.loads(pdtb_line)
            tokens, _, _ = preprocess(rel)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)


def load_relations(data_file):
    """Loads a single `relations.json` file

  Args:
    data_file: str, path to a single data file

  Returns:
    rels word embeddings
    labels a list of encoded labels
  """
    rel_path = os.path.join(data_file, "relations.json")
    assert os.path.exists(rel_path), \
        "{} does not exist in `load_relations.py".format(rel_path)

    instances = []
    labels = []

    with codecs.open(rel_path, encoding='utf-8') as pdtb:
        for pdtb_line in pdtb:
            rel = json.loads(pdtb_line)
            tokens, label, outputs = preprocess(rel)
            instances.append(tokens)
            labels.append(label)
            if "test" in data_file:
                output_rels.append(outputs)
    return labels, instances


def load_data(data_dir='./data'):
    """Loads all data in `data_dir` as a dict

  Each of `dev`, `train` and `test` contains (1) `raw` folder (2)
    `relations.json`. We don't need to worry about `raw` folder, and instead
    focus on `relations.json` which contains all the information we need for our
    classification task.

  Args:
    data_dir: str, the root directory of all data

  Returns:
    dict, where the keys are: `dev`, `train` and `test` and the values are lists
      of relations data in `relations.json`
  """
    assert os.path.exists(data_dir), "`data_dir` does not exist in `load_data`"

    data = {}
    label_dict = {}

    # get vocab
    print("Get vocabulary")
    folder_path = os.path.join(data_dir, "train")
    get_vocab(os.path.join(folder_path))

    for folder in os.listdir(data_dir):
        print("Loading", folder)
        folder_path = os.path.join(data_dir, folder)
        labels, instances = load_relations(folder_path)
        encoded_docs = np.array(featurize(instances))
        label_dict[folder] = np.array(labels)
        data[folder] = encoded_docs
    return data, label_dict, vocab


def featurize(instances: list) -> list:
    encoded_docs = []
    for instance in instances:
        encoded_doc = []
        for token in instance:
            if token in vocab:
                encoded_doc.append(vocab[token])
            else:
                # unknown words is assigned 0 arbitrarily
                encoded_doc.append(0)
        encoded_docs.append(encoded_doc)
    return encoded_docs

if __name__ == "__main__":
    data = load_data()
    print(data)
