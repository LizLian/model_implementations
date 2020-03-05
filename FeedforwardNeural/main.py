#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Entry point for the Program Assignment 2

Feel free to change/restructure the code below
"""
from model import DRSClassifier
from preprocessing import load_data
import numpy as np

# experiment parameters
units_1st = [512]
units_2nd = [128]
embedding_dims = [16]


def run_scorer(preds_file):
    """Automatically runs `scorer.py` on model predictions

  TODO: You don't need to use this code if you'd rather run `scorer.py`
    manually.

  Args:
    preds_file: str, path to model's prediction file
  """
    import os
    import sys
    import subprocess

    if not os.path.exists(preds_file):
        print(
            "[!] Preds file `{}` doesn't exist in `run_scorer.py`".format(preds_file))
        sys.exit(-1)

    python = 'python3.6'  # TODO: change this to your python command
    scorer = './scorer.py'
    gold = './data/test/relations.json'
    auto = preds_file
    command = "{} {} {} {}".format(python, scorer, gold, auto)

    print("Running scorer with command:", command)
    proc = subprocess.Popen(
        command, stdout=sys.stdout, stderr=sys.stderr, shell=True,
        universal_newlines=True
    )
    proc.wait()


def main():
    # loads and preprocesses data. See `preprocessing.py`
    data, labels, vocabs = load_data(data_dir='./data')

    # get embedding
    # embedding = get_embedding(vocabs)
    # trains a classifier on `train` and `dev` set. See `model.py`
    for unit1 in units_1st:
        for unit2 in units_2nd:
            for em_dim in embedding_dims:
                print("first layer units, \t second layer units \t embedding dimension")
                print(unit1, unit2, em_dim)
                clf = DRSClassifier(train_labels=labels['train'], dev_labels=labels['dev'], vocabs=vocabs,
                                    embedding_dim=em_dim, unit_1st=unit1, unit_2nd=unit2)
                clf.train(train_instances=data['train'], dev_instances=data['dev'])

                # output model predictions on `test` set
                preds_file = "./preds.json"
                clf.predict(data['test'], export_file=preds_file)

                # measure the accuracy of model predictions using `scorer.py`
                run_scorer(preds_file)


def load_word_embedding():
    embeddings_index = dict()
    with open("./glove/glove.6B.100d.txt") as f:
        lines = f.readlines()
    for line in lines:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    return embeddings_index


def get_embedding(vocabs):
    embeddings_index = load_word_embedding()

    embedding = {}
    # construct embedding_matrix
    embedding_matrix = np.zeros((len(vocabs["train"]), 100))
    for word in vocabs["train"]:
        if word in embeddings_index:
            embedding_matrix[vocabs["train"][word], :] = embeddings_index[word]
    unknown_vector = np.mean(embedding_matrix, axis=0)
    embedding["train"] = embedding_matrix

    for folder in vocabs:
        if folder != "train":
            embedding_matrix = np.zeros((len(vocabs[folder]), 100))
            for word in vocabs[folder]:
                if word in vocabs["train"]:
                    embedding_vector = embedding["train"][vocabs["train"][word]]
                    embedding_matrix[vocabs[folder][word]] = embedding_vector
                else: # unkown words
                    embedding_matrix[vocabs[folder][word]] = unknown_vector
            embedding[folder] = embedding_matrix
    return embedding


if __name__ == '__main__':
    main()
