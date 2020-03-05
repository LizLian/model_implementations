# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import numpy as np
import scipy.misc
import pickle

MAX_ITER = 100
DATA_SET = 3
LAMBDA = 0.5

class MaxEnt(Classifier):

    def __init__(self):
        self.vocab = {}
        self.labels = {}
        self.weights = None
        self.max_accuracy = 0

    def get_model(self):
        return None

    def set_model(self, model):
        pass

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, 0.0001, 10)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient 
        """
        f = open("result1000.txt", 'a')
        f.write("Training size: " + str(len(train_instances)) + ", " + "Batch size: " + str(batch_size) + "\n")
        self._set_feature_vector(train_instances)
        # weights_train = self.set_feature_vector(train_instances)

        # initialize all weights to 1
        # bias is the last feature
        self.weights = np.zeros((len(self.labels), len(self.vocab) + 1))
        self.weights[:, len(self.vocab)] = np.ones(len(self.labels))

        j = 0
        while j < MAX_ITER:
            p_labels = np.zeros(len(self.labels))
            i = 1
            total_instances = 1
            features = np.zeros((len(self.labels), len(self.vocab) + 1))
            true_labels = np.zeros((len(self.labels), len(self.vocab) + 1))
            gradients = np.zeros((len(self.labels), len(self.vocab) + 1))
            loss = 0
            for instance in train_instances:
                if i <= batch_size and total_instances <= len(train_instances):
                    features = features + instance.feature_vector
                    true_labels[self.labels[instance.label], :] = instance.feature_vector
                    # update the value to 1 if the features present
                    # true_labels = (true_labels > 0).astype(int)
                    for label in self.labels:
                        n = np.dot(instance.feature_vector, self.weights[self.labels[label], :])
                        d = np.dot(self.weights, instance.feature_vector)
                        p_labels[self.labels[label]] += np.exp(n - scipy.misc.logsumexp(d))
                                                               #np.log(np.sum(np.exp(d))))

                    i += 1
                    total_instances += 1
                    posterior = np.transpose(np.transpose(features) * p_labels)
                    gradients += posterior - true_labels

                    # set loss function
                    # posterior is y_hat, true_labels is y
                    loss -= np.sum(np.dot(self.weights, instance.feature_vector)) \
                            - scipy.misc.logsumexp(np.dot(self.weights, np.transpose(posterior)))

                    # reset the variables
                    features = np.zeros((len(self.labels), len(self.vocab) + 1))
                    true_labels = np.zeros((len(self.labels), len(self.vocab) + 1))
                    p_labels = np.zeros(len(self.labels))
                else:
                    gradients += LAMBDA * self.weights
                    # update weights
                    self.weights -= learning_rate * gradients
                    gradients = np.zeros((len(self.labels), len(self.vocab) + 1))

                    i = 1

            accuracy = self._converge(dev_instances)
            print("Iteration " + str(j) + ": ", accuracy)
            write_to_file = "Iteration " + str(j) + ": " + str(accuracy)
            f.write(write_to_file + "\n")
            # save the parameters
            #self._save_model(accuracy, j)
            j += 1
            # print total loss
            # print("Iteration " + str(j) + ": ", round(loss, 4), accuracy)
        f.close()

    def _save_model(self, accuracy, iteration):
        if accuracy > self.max_accuracy:
            model_file = open("finalized_model_" + str(DATA_SET) + "_" + str(iteration) + "_" + str(accuracy) + ".sav", "wb")
            self.save(model_file)
            self.max_accuracy = accuracy
            model_file.close()

    def _load_model(self, model_name):
        self.load(model_name)
        return self.model

    def _converge(self, dev_instances):
        """
        use the dev data set for converge
        :param dev_instances: dev data
        :return: accuracy rate
        """
        correct = [self.classify(x) == x.label for x in dev_instances]
        return round(float(sum(correct)) / len(correct), 4)

    def _set_feature_vector(self, data):
        """
        Set feature vector for each document instance
        :param data: instances of documents
        :return: an empty n by m matrix, n being the number of labels and m being the number of features
        """
        i = 0
        j = 0
        for instance in data:
            if instance.label not in self.labels:
                self.labels[instance.label] = j
                j += 1
            features = instance.features()
            for feature in features:
                if feature not in self.vocab:
                    self.vocab[feature] = i
                    i += 1

        for instance in data:
            feature_vector = np.zeros(len(self.vocab) + 1)
            feature_vector[len(self.vocab)] = 1  # bias feature
            features = instance.features()
            for feature in features:
                feature_vector[self.vocab[feature]] += 1
            instance.feature_vector = feature_vector

    def classify(self, instance):
        features = instance.features()
        feature_vector = np.zeros(len(self.vocab) + 1)
        feature_vector[len(self.vocab)] = 1  # bias feature
        instance.feature_vector = feature_vector
        for feature in features:
            if feature in self.vocab:
                feature_vector[self.vocab[feature]] += 1
            instance.feature_vector = feature_vector
        labels = {self.labels[label]: label for label in self.labels}
        return labels[np.argmax(np.dot(self.weights, instance.feature_vector))]

if __name__=="__main__":
    finalized_model = ""
