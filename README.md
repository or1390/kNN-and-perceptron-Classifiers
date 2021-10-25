# Mini-Project 1 - kNN and Perception algorithms
#### This project consists on implementation of two machine learning algorithms: kNN and Perception.
#### They can be used both for binary and multiclass classification. The aim was to implement them from scratch, without importing libraries which already have implemented those. 

#### In the main file, we have imported the files for each algorithm: kNN and Perception. 
#### In the main.py file, we do:
* Data reading and preprocessing
	1. Read the dataset.
	2. Shuffle it to pick up a random training sample. Also, a random test size of 0.3.
	3. In case the dataset is large( ex. a4a), we set a subset size to run it faster. 
* kNN-algorithm
	* ##### It is designed to work with two types of distances: Eucledian and Manhattan
	* ##### The number of Nearest Neighbors is parametrized. 
	* ##### There is no training process before testing the model, because the concept of KNN is to find the nearest data points to the required data.
	* ##### The accuracy is calculated as the number of correctly predicted data points(labels) out of all the data points(labels).

* Perception
  * ##### It takes into consideration the number of iterations and the classes(labels). The labels are passed a list, so it can be used for binary or multiclass classification problems.
  * ##### We do train and test our model.
  * ##### The accuracy is calculated as the number of correctly predicted data points(labels) out of all the data points(labels).


#### In the m1_kNN.py file:
  * ##### The implementation of kNN algorithm which can work using Eucledian and Manhattan distances


#### In the m2_perception.py file:
  * ##### The implementation of Perception. In addition, we have implemented a function to plot the feature importance for both datasets: a4a( binary classification) and iris ( multiclass classification).


# Find below the results:

![image](https://github.com/or1390/mini-project1/blob/f345bd271af99d7987a98be3c1d477c633ccdf13/iris_dataset_kNN_1.png)<br/>
![image](https://github.com/or1390/mini-project1/blob/f345bd271af99d7987a98be3c1d477c633ccdf13/a4a_dataset_kNN_2.png)<br/>
![image](https://github.com/or1390/mini-project1/blob/f345bd271af99d7987a98be3c1d477c633ccdf13/a4a_dataset_perception_1.png)<br/>
![image](https://github.com/or1390/mini-project1/blob/f345bd271af99d7987a98be3c1d477c633ccdf13/iris_dataset_perception_2.png)<br/>

# Feature Importance for Perception
### a4a dataset - Binary classification
![image](https://github.com/or1390/mini-project1/blob/f345bd271af99d7987a98be3c1d477c633ccdf13/a4a_feature_importance_perception.png)
### iris dataset - Multiclass classification
![image](https://github.com/or1390/mini-project1/blob/f345bd271af99d7987a98be3c1d477c633ccdf13/iris_feature_importance_perception.png)



	
	



