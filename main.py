'''
This is the main function which basically reads the datasets,
preprocess by splitting, shuffling them and taking a sample
subset for
the a4a dataset.
Also, I have imported to algorithms KNN and perception
which are executed for both datasets.
'''
from m1_kNN import KNN
from m2_perception import perception
import numpy as np
import random
from sklearn.datasets import load_svmlight_file
################a4a Dataset###############
df = load_svmlight_file("a4a")
X = df[0]
y = df[1]
new_X = X.toarray()
new_y = np.round(y).astype(int)
################Iris Scale Dataset###########
#df = load_svmlight_file("iris.scale")
#X = df[0]
#y = df[1]
#new_X = X.toarray()
#new_y = np.round(y).astype(int)
####Shuffling #################
shuffler = np.random.permutation(len(new_X))
shuffled_x = new_X[shuffler]
shuffled_y = new_y[shuffler]
random.shuffle(new_X)
random.shuffle(new_y)
test_size = 0.3
training_count = len(shuffled_x) - len(shuffled_x) * test_size
X_train = shuffled_x[:int(training_count)]
X_test = shuffled_x[int(training_count):]
y_train = shuffled_y[:int(training_count)]
y_test = shuffled_y[int(training_count):]
##########Subset#################
subset_size = 500
X_train = X_train[0:subset_size,]
y_train = y_train[0:subset_size]
X_test = X_test[0:subset_size,]
y_test = y_test[0:subset_size]


###KNN -algorithm
k= 3
knn = KNN(k, "euclidean")
y_prediction= knn.predict(X_train, y_train, X_test)
accuracy = np.sum(y_prediction == y_test)/ len(y_test)
print("Accuracy for KNN is:\n" , accuracy)
####Perceptron -algorithm
'''
prc = perception(1000, [-1, 1]) #a4a dataset
prc = perception(1000, [1,2,3]) #iris dataset
feature_weight = prc.train(X_train, y_train)
y_prediction = prc.predict(X_test)
prc.feature_importance()
accuracy = np.sum(y_prediction == y_test) / len(y_test)
print("Accuracy for Perception is:\n", accuracy)
'''
