'''
The implementation of the Perception algorithm is adapted to
work with multiclass classification.
The main functions are train which calls
get_multiclass_outcome function and predict.
To implement the visualization of feature importance is
defined feature_importance function which is executed
for specific conditions of binary classification and multiclass
classification
'''
import numpy as np
from matplotlib import pyplot
class perception:
    #initialize method
    def __init__(self, iterations, classes):
        self.iterations = iterations
        self.weights = {}
        self.classes = classes
        self.n_classes = len(classes)
    ## multiclass decision rule
    def get_multiclass_outcome(self, x):
        get_max_class = 0
        get_max_value = 0
        for c in self.classes:
        #output = w*x
            output = np.dot(self.weights[str(c)], x)
            if (output >= get_max_value):
                get_max_value = output
                get_max_class = c
        return get_max_class
    def train(self, X_train, y_train):
        samples, features = X_train.shape
        ## initiliaze weights, all zero
        for i_class in self.classes:
            self.weights[str(i_class)] = np.zeros(features)
            #loop through all the iterations
        for it in range(self.iterations):
            #loop through training data
            for indx_x, x_i in enumerate(X_train):
            # w*x +b
                y_label = self.get_multiclass_outcome(x_i)
                actual = y_train[indx_x]
            #if the predicted value is wrong, the weight vectors are corrected
            if (y_label != actual):
            ##Update weights rule
                self.weights[str(actual)] =  self.weights[str(actual)] + x_i
                self.weights[str(y_label)] = self.weights[str(y_label)] -x_i
        return self.weights

    # predict using the weights
    def predict(self, X_test):
        p = []
        #loop throught the testing data
        for indx_x, x_i in enumerate(X_test):
            y_class = self.get_multiclass_outcome(x_i)
            p.append(y_class)
        return p
    ### visualize the importance of the features
    def feature_importance(self):
        ###binary classification
        fig, axs = pyplot.subplots(2, 1)
        features = np.arange(122)
        axs[0].bar(features, np.abs(self.weights["-1"]))
        axs[0].set_title("Class 1")
        axs[1].bar(features, np.abs(self.weights["1"]))
        axs[0].sharex(axs[0])
        axs[1].set_title("Class 2")
        fig.tight_layout()
        pyplot.show()
        '''
        ###multiclass classification
        fig, axs = pyplot.subplots(3, 1)
        features = np.arange(4)
        axs[0].bar(features, np.abs(self.weights["1"]))
        axs[0].set_title("Class 1")
        axs[1].bar(features, np.abs(self.weights["2"]))
        axs[0].sharex(axs[0])
        axs[1].set_title("Class 2")
        axs[2].bar(features, np.abs(self.weights["3"]))
        axs[0].sharex(axs[0])
        axs[2].set_title("Class 3")
        fig.tight_layout()
        pyplot.show()
        '''

