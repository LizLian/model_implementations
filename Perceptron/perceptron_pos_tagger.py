import numpy as np

import train_test_tagger, time
from data_structures import Sentence
from collections import defaultdict

class Perceptron_POS_Tagger(object):
    def __init__(self, algorithm="reg"):
        ''' Modify if necessary. 
        '''
        self.tags = {0:f"{'pos-1:START'}"} # {0: START, 1: DT, 2: NN, ...}
        self.tags_reversed = {f"{'pos-1:START'}":0} # {START: 0, DT: 1, NN: 2, ...}
        # self.weights = {}
        self.weights = defaultdict(np.zeros((1,46)))
        self.n = 0
        self.step = 0
        self.t = 0
        self.algorithm = algorithm.lower() # "avg" for average perceptron. "reg" for regular perceptron
        # feature for ablation study
        self.feat_exp = ["word2", "word1", "word0", "word-1", "word-2", "prefix", "suffix", "pos-1"]
        self.no_prev_tag = False

    def viterbi(self, instance, features_for_ablation):
        ''' Implement the Viterbi decoding algorithm here.
        '''
        feat = Sentence(instance)
        transition = np.zeros((len(self.tags), len(self.tags)))
        v_matrix = np.zeros((len(self.tags), len(instance.snt)))
        b_matrix = np.zeros((len(self.tags), len(instance.snt)))  # backpointer
        local_scores = np.zeros((1, len(self.tags)))
        # first position
        # the reason that we don't use the feature_set in the sentence object is that we don't
        # want to include the prev-tag feature here, not until after viterbi algorithm
        features = feat.features(instance.snt, 0, features_for_ablation)
        for feature in features:
            if feature in self.weights:
                local_scores += self.weights[feature]
        local_scores += self.weights[f"{'pos-1:START'}"]
        v_matrix[:, 0] = np.reshape(np.transpose(local_scores), (len(self.tags), ))
        for pos in range(1, len(instance.snt)):
            local_scores = np.zeros((1, len(self.tags)))
            features = feat.features(instance.snt, pos, features_for_ablation)

            for tag_pos in self.tags:
                transition[:, tag_pos] = np.reshape(np.transpose(self.weights[f"pos-1:{self.tags[tag_pos]}"]), (len(self.tags), ))

            for feature in features:
                if feature in self.weights:
                    local_scores += self.weights[feature]
            v_matrix[:, pos] = np.max((transition + np.transpose(local_scores) + v_matrix[:, pos-1]), axis=1)
            b_matrix[:, pos] = np.argmax((transition + np.transpose(local_scores) + v_matrix[:, pos-1]), axis=1)

        pointer = np.argmax(v_matrix[:, -1])
        best_path = [pointer]
        for t in reversed(range(1, len(instance.snt))):
            pointer = int(b_matrix[pointer][t])
            best_path.insert(0, pointer)
        return best_path

    def tag(self, test_data, features_for_ablation):
        predicted_tags = []
        for sent in test_data:
            best_path = self.viterbi(sent, features_for_ablation)
            predicted_tags.append([self.tags[path] for path in best_path])
        return predicted_tags

    def train(self, train_data, features_for_ablation):
        ''' Implement the Perceptron training algorithm here.
        @param train_data training data set
        '''
        for sent in train_data:
            self.t += 1
            best_path = self.viterbi(sent, features_for_ablation)
            for i in range(len(sent.snt)):
                # add previous tag to the feature list so these features can be updated in weights
                if not self.no_prev_tag:
                    if i == 0:
                        sent.feature_set[0].append(f"{'pos-1:START'}")
                    else:
                        sent.feature_set[i].append(f"pos-1:{self.tags[best_path[i-1]]}")

                predicted_tag = self.tags[best_path[i]]
                if sent.correct_tags[i] != predicted_tag:
                    for feature in sent.feature_set[i]:
                        if self.algorithm.lower() == "reg":
                            self.weights[feature][:, self.tags_reversed[sent.correct_tags[i]]] += 1
                            self.weights[feature][:, self.tags_reversed[predicted_tag]] -= 1
                        else:
                            self.weights[feature][:, self.tags_reversed[sent.correct_tags[i]]] += self.step/self.n
                            self.weights[feature][:, self.tags_reversed[predicted_tag]] -= self.step/self.n
            self.step -= 1

            # print progress bar
            if self.t == 1000:
                print(".", end="", flush=True)
                self.t = 0

    def accuracy(self, gold_tags, pred_tags):
        """
        calculate the accuracy of the model
        return the accuracy in decimals
        :param gold_tags: a list of a sequence of correct tags
        :param pred_tags: a list of a sequence of incorrect tags
        :return: accuracy in decimals
        """
        correct = 0.0
        total = 0.0
        for g_snt, pred_snt in zip(gold_tags, pred_tags):
            correct += sum([gold_tag == pred_tag for gold_tag, pred_tag in zip(g_snt, pred_snt)])
            total += len(g_snt)
        return correct / total

    def _initialize_model(self, train_data, features_for_ablation):
        """
        initialize model paramters including pos tags, feature dictionary
        add feature set, gold pos tags to sentence obj
        initialize weight vector for each feature, including all possible previous tags
        :param train_data: train data set
        """
        all_features = []
        for sent in train_data:
            sentence = Sentence(sent)
            correct_tags = []
            for position in range(len(sent.snt)):
                token = sent.snt[position]
                # get pos tags for training data
                if token[1] not in self.tags_reversed:
                    self.tags[len(self.tags)] = token[1]
                    self.tags_reversed[token[1]] = len(self.tags_reversed)

                # build feature dictionary. add features to each sentence obj.
                features = sentence.features(sent.snt, position, features_for_ablation)
                sent.feature_set[position] = features
                all_features += features

                # get correct tags
                correct_tags.append(token[1])
            sent.correct_tags = correct_tags

        # initialize weights for all features in training data
        self.weights = {feature: np.zeros((1, len(self.tags))) for feature in set(all_features)}

        # initialize weights for all prev pos tags
        for tag in self.tags_reversed:
            self.weights[f"pos-1:{tag}"] = np.zeros((1, len(self.tags)))
        self.weights[f"{'pos-1:START'}"] = np.zeros((1, len(self.tags)))


    def run(self, train_data, gold_dev_data, plain_dev_data, epoch=10, n=50):
        """
        this method is to run the model on the full feature set
        @:param train_data training data set
        @:param gold_dev_data the dev data that contains gold tags
        @:param plain_dev_data the dev set that doesn't contain gold standards
        @:param epoch number of epochs that you want to train the model
        @:param n batch size used in averaged perceptron, default to 50
        """
        start = time.time()
        self.n = len(train_data)
        self.step = self.n
        # initialize model paramters (see the method)
        self._initialize_model(train_data, self.feat_exp)

        # create batches
        batch_data = []
        if self.algorithm == "avg":
            i = 0
            while i < self.n:
                batch_data.append(train_data[i:i + n])
                i += n

        # get dev gold tags for accuracy
        dev_gold_tags = []
        for sent in gold_dev_data:
            gold_tags = [token[1] for token in sent.snt]
            dev_gold_tags.append(gold_tags)

        for e in range(epoch):
            if self.algorithm == "avg":
                for batch in batch_data:
                    self.train(batch, self.feat_exp)
            else:
                self.train(train_data, self.feat_exp)
            print(".")

        # predict and evaluate
        predicted_tags = self.tag(plain_dev_data, self.feat_exp)
        end = time.time()
        print(f"Total time: {round(end - start, 0)}s", flush=True)
        return predicted_tags


    def main(self, train_data, gold_dev_data, plain_dev_data, epoch=10, n=50):
        """
        this method is for ablation study
        calls method _initialize_model(), train(), and accuracy()
        applies average perceptron or regular perceptron
        prints results per epoch
        @:param train_data training data set
        @:param gold_dev_data the dev data that contains gold tags
        @:param plain_dev_data the dev set that doesn't contain gold standards
        @:param epoch number of epochs that you want to train the model
        @:param n batch size used in averaged perceptron, default to 50
        """
        # get all features for ablation study
        features_for_ablation = self.get_features_ablation()
        for f in features_for_ablation:
            if f == 7: # drop previous tag
                self.no_prev_tag = True
            feats_for_ablation = features_for_ablation[f]
            print(feats_for_ablation)
            start = time.time()
            self.n = len(train_data)
            self.step = self.n

            # initialize model paramters (see the method)
            self._initialize_model(train_data, feats_for_ablation)

            # create batches
            batch_data = []
            if self.algorithm == "avg":
                i = 0
                while i < self.n:
                    batch_data.append(train_data[i:i+n])
                    i += n

            # get dev gold tags for accuracy
            dev_gold_tags = []
            for sent in gold_dev_data:
                gold_tags = [token[1] for token in sent.snt]
                dev_gold_tags.append(gold_tags)

            for e in range(epoch):
                if self.algorithm == "avg":
                    for batch in batch_data:
                        self.train(batch, feats_for_ablation)
                else:
                    self.train(train_data, feats_for_ablation)
                print(".")

                # predict and evaluate
                predicted_tags = self.tag(plain_dev_data, feats_for_ablation)
                accuracy = self.accuracy(dev_gold_tags, predicted_tags)
                end = time.time()
                print(f"Epoch: {e} \t Total time: {round(end - start, 0)}s \t Accuracy: {round(accuracy, 5)}", flush=True)
                start = end
                self.step = len(train_data)
            self.no_prev_tag = False

    def get_features_ablation(self):
        # build features for ablation study
        feat_study = {}
        for i in range(8):
            feat_study[i] = [self.feat_exp[j] for j in range(8) if j!=i]
        feat_study[8] = ["word2", "word1", "word0", "word-1", "word-2", "prefix", "suffix", "pos-1"]
        return feat_study

if __name__ == "__main__":
    train_file = "./train/ptb_02-21.tagged"
    gold_dev_file = "./dev/ptb_22.tagged"
    plain_dev_file = "./dev/ptb_22.snt"
    test_file = "./test/ptb_23.snt"

    # Read in data
    train_data = train_test_tagger.read_in_gold_data(train_file)
    gold_dev_data = train_test_tagger.read_in_gold_data(gold_dev_file)
    plain_dev_data = train_test_tagger.read_in_plain_data(plain_dev_file)
    # test_data = train_test_tagger.read_in_plain_data(test_file)

    # regular Perceptron
    pos_tagger = Perceptron_POS_Tagger()
    pos_tagger.main(train_data, gold_dev_data, plain_dev_data, epoch=3)

    # averaged Perceptron
    avg_pos_tagger = Perceptron_POS_Tagger("avg")
    avg_pos_tagger.main(train_data, gold_dev_data, plain_dev_data, epoch=3, n=50)
