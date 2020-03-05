import numpy as np
from preprocessing import MAX_LEN_CON, MAX_LEN_1, MAX_LEN_2, label_ref, output_rels, load_data, vocab


class Layer:
    def __init__(self, units, activation_function=None):
        self.units = units
        self.activation_function = activation_function
        self.activations = None
        self.params = None
        self.gradient = None
        # network direction = forward
        self.prevLayer = None
        self.nextLayer = None
        self.isInputLayer = 0
        self.isOutputLayer = 0


class FeedForwardNetwork:
    def __init__(self):
        self.layerSettings = []
        self.act_func = {"relu": self._relu,
                         "softmax": self._softmax,
                         "sigmoid": self._sigmoid}
        self.act_func_deriv = {"relu": self._relu_deriv,
                         "sigmoid": self._sigmoid_deriv}

    def add(self, layer: Layer):
        self.layerSettings.append(layer)

    def train(self, train_instances, train_labels, batch=50, epoch=1):
        # link layers
        for i in range(0, len(self.layerSettings)-1):
            curr_layer = self.layerSettings[i]
            next_layer = self.layerSettings[i+1]
            curr_layer.nextLayer = next_layer
            next_layer.prevLayer = curr_layer
            next_layer.gradient = np.zeros((next_layer.units, curr_layer.units))
        self.layerSettings[-1].prevLayer = curr_layer

        total_epoch = epoch
        while epoch > 0:
            counter = 0
            print("Epoch " + str(epoch) + "/" + str(total_epoch))
            epoch -= 1
            while counter < len(train_instances):
                counter += batch
                batch_instances = train_instances[counter-batch: counter]
                inputLayer = self.layerSettings[0]
                inputLayer.isInputLayer = 1
                # input one-hot vector
                inputLayer.activations = self._get_input_matrix(batch_instances, inputLayer.units)

                curr_layer = inputLayer.nextLayer
                while curr_layer is not None:
                    weights = np.random.randint(2, size=(curr_layer.units, curr_layer.prevLayer.units))
                    # assign the bias term to one
                    weights[:, 0] = np.ones(curr_layer.units)
                    s = np.dot(curr_layer.prevLayer.activations, np.transpose(weights))
                    curr_layer.activations = self.act_func[curr_layer.activation_function](s)
                    curr_layer.params = weights.astype(np.float)
                    curr_layer = curr_layer.nextLayer

                outputLayer = self.layerSettings[-1]
                outputLayer.isOutputLayer = 1
                y_hat = outputLayer.activations

                # back propagation
                y = self._get_output_matrix(train_labels[counter-batch: counter])
                D = y_hat - y
                prev_out = D

                # update output layer gradients
                outputLayer.gradient = np.transpose(np.dot(np.transpose(outputLayer.prevLayer.activations), D))

                curr_layer = outputLayer.prevLayer
                while curr_layer is not None and curr_layer.isInputLayer != 1:
                    C = np.dot(prev_out, curr_layer.nextLayer.params)
                    deriv = self.act_func_deriv[curr_layer.activation_function](curr_layer.activations)
                    D = np.multiply(deriv, C)
                    prev_out = D
                    curr_layer.gradient = np.transpose(np.dot(np.transpose(curr_layer.prevLayer.activations), D))
                    curr_layer.params -= curr_layer.gradient
                    # set the gradients back to 0s
                    # curr_layer.gradient = np.zeros((curr_layer.units, curr_layer.prevLayer.units))
                    curr_layer = curr_layer.prevLayer

    def predict(self, instances, batch=50):
        preds = []
        inputLayer = self.layerSettings[0]
        counter = 0
        while counter < len(instances):
            counter += batch
            batch_instances = instances[counter-batch: counter]
            inputLayer.activations = self._get_input_matrix(batch_instances, inputLayer.units)
            curr_layer = inputLayer.nextLayer
            while curr_layer is not None:
                s = np.dot(curr_layer.prevLayer.activations, np.transpose(curr_layer.params))
                curr_layer.activations = self.act_func[curr_layer.activation_function](s)
                curr_layer = curr_layer.nextLayer
            outputLayer = self.layerSettings[-1]
            y_hat = outputLayer.activations
            preds.append(np.argmax(y_hat))
        return preds

    def _sigmoid(self, x):
        return (1/(1+np.exp(-x)))

    def _sigmoid_deriv(self, z):
        return z*(1-z)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, z):
        z[z<=0] = 0
        z[z>0] = 1
        return z

    def _softmax(self, X, theta=1.0, axis=1):
        """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
            first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """
        # make X at least 2d
        y = np.atleast_2d(X)
        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
        # multiply y against the theta parameter,
        y = y * float(theta)
        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)
        # exponentiate y
        y = np.exp(y)
        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
        # finally: divide elementwise
        p = y / ax_sum
        # flatten if X was 1D
        if len(X.shape) == 1: p = p.flatten()
        return p

    def _get_input_matrix(self, batch_instances, c):
        matrix = np.zeros((len(batch_instances), c))
        for i in range(len(batch_instances)):
            for index in batch_instances[i]:
                matrix[i, index] += 1
        return matrix

    def _get_output_matrix(self, train_labels):
        matrix = np.zeros((len(train_labels), len(label_ref)))
        for i in range(len(train_labels)):
            matrix[i, train_labels[i]] = 1
        return matrix


if __name__=="__main__":
    # loads and preprocesses data. See `preprocessing.py`
    data, labels, vocabs = load_data(data_dir='./data')
    train_instances = data["train"]
    inputLayer = Layer(len(vocabs))
    hiddenLayer1 = Layer(512, "relu")
    hiddenLayer2 = Layer(256, "relu")
    outputLayer = Layer(len(label_ref), "softmax")
    FFN_model = FeedForwardNetwork()
    FFN_model.add(inputLayer)
    FFN_model.add(hiddenLayer1)
    FFN_model.add(hiddenLayer2)
    FFN_model.add(outputLayer)
    FFN_model.train(data["train"], labels["train"])
    preds = FFN_model.predict(data["test"])
    print(preds)

