'''
The implementation of the KNN algorithm by defining
two distances: Euclidean and Manhattan distance as global functions
The class is initialized by defining the k- value, which represents
the k-nearest neighbors and the method applied to calculate the distance.
Then, the predict method calculates the predicted labels for the testing
dataset.
'''
import numpy as np
from scipy.stats import mode
#######Global Functions to calculate the distance #######
def eucledian(x1,x2):
    for i in range(len(x1) - 1):
        distance= np.sqrt(np.sum((x1-x2)**2))
    return distance
def mahnattan(x1,x2):
    for i in range(len(x1) - 1):
        distance= np.sum(np.abs((x1-x2)))
    return distance
########################################################
class KNN:
    def __init__(self,k, method):
        self.k = k
        self.method = method
    # Function to calculate KNN
    def predict(self, x_train, y, x_test):
        y_labels = []
        # Loop through the testing data
        for i_test in x_test:
            output_of_distances = []
        # Loop through each training Data
            for j_train in range(len(x_train)):
            # Calculate the distance by using Eucledian or Manhattan distance
                if self.method == "euclidean":
                    distance = eucledian(np.array(x_train[j_train, :]), i_test)
                else:
                    distance = mahnattan(np.array(x_train[j_train, :]), i_test)
                # Storing the distances
                output_of_distances.append(distance)
            output_of_distances = np.array(output_of_distances)
            # Sorting the array and keeping the first K datapoints
            sorted_distances = np.argsort(output_of_distances)[:self.k]
            # Labels of the K datapoints from above
            labels = y[sorted_distances]
            # Majority voting
            lab = mode(labels)
            lab = lab.mode[0]
            y_labels.append(lab)
        return y_labels
