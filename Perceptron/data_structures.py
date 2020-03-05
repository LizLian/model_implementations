class Sentence(object):
    def __init__(self, snt):
        ''' Modify if necessary.
        '''
        self.snt = snt
        self.window = 2
        self.feature_set = {}
        self.correct_tags = []

    def features(self, sent, position, features_for_ablation):
        ''' Implement your feature extraction code here. This takes annotated or unannotated sentence
        and return a set of features
        '''
        feature_set = []
        for i in range(self.window, -1-self.window, -1):
            if position+i < 0 or position+i >= len(sent):
                pass
            else:
                if type(sent[position+i]) == str:
                    feature_set.append(f"word{i}:{sent[position+i]}")
                else:
                    feature_set.append(f"word{i}:{sent[position+i][0]}")

        if type(sent[position]) == str:
            feature_set.append(f"prefix:{sent[position][0:2]}")
            feature_set.append(f"suffix:{sent[position][-2:]}")
        else:
            feature_set.append(f"prefix:{sent[position][0][0:2]}")
            feature_set.append(f"suffix:{sent[position][0][-2:]}")
        feature_set = [feature for feature in feature_set if feature.split(":", 1)[0] in features_for_ablation]
        self.feature_set[position] = feature_set
        return feature_set
