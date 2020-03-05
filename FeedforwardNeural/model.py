#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Discourse Relation Sense Classifier

Feel free to change/restructure the code below
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
from keras.utils.np_utils import to_categorical
from preprocessing import MAX_LEN_CON, MAX_LEN_1, MAX_LEN_2, label_ref, output_rels
import numpy as np
import json


class DRSClassifier(object):
    """Implement a FeedForward Neural Network for Discourse Relation Sense
      Classification using Tensorflow/Keras (tensorflow 2.0)"""

    def __init__(self, train_labels, dev_labels, vocabs, embedding_dim, unit_1st, unit_2nd):
        self.output_size = len(set(train_labels))
        self.train_labels = to_categorical(train_labels)
        self.dev_labels = to_categorical(dev_labels, num_classes=self.output_size)
        self.vocabs = vocabs
        # self.embedding = embedding
        self.model = self.build(embedding_dim=embedding_dim, unit_1st=unit_1st, unit_2nd=unit_2nd)

    def build(self, embedding_dim, unit_1st, unit_2nd):
        model = Sequential()
        model.add(Embedding(len(self.vocabs), embedding_dim, input_length=MAX_LEN_1+MAX_LEN_2+MAX_LEN_CON))
        model.add(Conv1D(256, 50, activation="relu"))
        model.add(GlobalMaxPooling1D())
        model.add(Flatten())
        model.add(Dense(unit_1st, activation="relu"))
        model.add(Dense(unit_2nd, activation="relu"))
        model.add(Dense(self.output_size, activation="softmax"))
        # compile the model
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        # print model summary
        print(model.summary())
        return model

    def train(self, train_instances, dev_instances):
        """Train the classifier on `train_instances` while evaluating
        periodically on `dev_instances`

        Args:
          train_instances: list
          dev_instances: list
        """
        self.model.fit(train_instances, self.train_labels, batch_size=64, epochs=5)
        # evaluate the model using dev set
        loss, accuracy = self.model.evaluate(dev_instances, self.dev_labels, batch_size=128)
        print("Dev Accuracy: %f" %(accuracy*100))

    def predict(self, instances, export_file="./preds.json"):
        """Given a trained model, make predictions on `instances` and export
        predictions to a json file

        Args:
          instances: list
          export_file: str, where to save your model's predictions on `instances`

        Returns:

        """
        predictions = np.array(self.model.predict(instances))
        preds = np.argmax(predictions, axis=1)
        reversed_label_ref = {label_ref[label]: label for label in label_ref}
        pred_labels = [reversed_label_ref[pred] for pred in preds]

        for i in range(len(output_rels)):
            output_rels[i]["Sense"] = [pred_labels[i]]

        with open(export_file, "w") as outfile:
            for rel in output_rels:
                json.dump(rel, outfile)
                outfile.write("\n")
