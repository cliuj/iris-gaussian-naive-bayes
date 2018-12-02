import numpy as np
import urllib.request
from random import shuffle
import math


# global variables used to for configuration
debug = False
has_dataset = True
website_hosting_dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset_file = "dataset.txt"
dataset = []

iris_dataset = []
training_set = []
test = [5.7, 2.8, 4.1, 1.3]


def print_objects_of(array):
    for element in array:
        element.to_string()
    print()


# The following both retrieves the UTF-8 encoded data and
# decodes the data into string data
def retrieve_dataset_from(website):
    raw_dataset = urllib.request.urlopen(website)
    dataset = []
    for byte_data in raw_dataset:
        str_data = byte_data.decode('utf-8').rstrip()
        dataset.append(str_data)
    return dataset


def write_dataset_to(output_file, dataset):
    dataset_file = open(output_file, 'w')
    for object_data in dataset:
        dataset_file.write("{}\n".format(object_data))


# Set global variable "has_dataset" to False if dataset.txt is missing or empty
# this way, the dataset is created and stored into dataset.txt
if (not has_dataset):
    dataset = retrieve_datatset_from(website_hosting_dataset)
    write_dataset_to(dataset_file, dataset)
    if(debug):
        for data in dataset:
            print("{}".format(data))


# Iris object to store the values
class Iris:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, iris_class):

        self.features = {
            'sepal_length' : float(sepal_length),
            'sepal_width' : float(sepal_width),
            'petal_length' : float(petal_length),
            'petal_width' : float(petal_width),
            'iris_class' : iris_class
        }

    def to_string(self):
        print("sepal_length: {}, sepal_width: {}, petal_length: {}, petal_width: {}, iris_class: {}"
            .format(self.features['sepal_length'],
                    self.features['sepal_width'],
                    self.features['petal_length'],
                    self.features['petal_width'],
                    self.features['iris_class']))


# Parse the data
def parse_file(to_parse_file):
    return [line.rstrip() for line in open(to_parse_file)]


def retrieve_data_from(dataset_file):
    parsed_file = parse_file(dataset_file)
    
    # for every iris data in dataset file
    # parse the data and pass it into a new
    # iris object which is then stored in a global
    # list of irises called "iris_dataset"
    for iris in parsed_file:
        iris_features = iris.split(',')

        sepal_length = iris_features[0]
        sepal_width = iris_features[1]
        petal_length = iris_features[2]
        petal_width = iris_features[3]
        iris_class = iris_features[4]

        iris_dataset.append(Iris(sepal_length, sepal_width, petal_length, petal_width, iris_class))


# Selects the Iris stored int the dataset based on the passed number
#
# Ex.
# draw_samples_for_training(25) will result in having 25/50 of the different
# irises choosen. So in our dataset of 150 total iris with 3 different class of 50 iris,
# 25 out of those 50 in each iris class (setosa, vericolor, virginica) will be selected for
# the training data
#
def draw_samples_for_training(class_sample_num):
    for iris_class in range(0, 3): 
        t = 0
        for i in range(iris_class * 50, (iris_class + 1) * 50):
            if t >= class_sample_num:
                break
            training_set.append(iris_dataset[t + (iris_class * 50) ])
            t += 1


# Take all the values of a feature in a certain section of the training_set and store them
# into a new array and then return that same array
#
# @parameter 'feature': pass a key of Iris.feature to return an array of all of those values
# @parameter 'section': determines the part of the training_set to extract the features 
#                       into the array:
#                       1 setosa
#                       2 versicolor
#                       3 virginica
#
# *Note that the numbers for the sections above is only if the classes are ordered this way
#  in the training_set; re-number them based on how it is orderd in the training_set
#
def isolated_feature(feature, section):
    start = ((section - 1) / 3) * len(training_set)
    end = (section / 3) * len(training_set)
    isolated_features_list = []
    for i in range(int(start), int(end)):
        isolated_features_list.append(training_set[i].features[feature])
    return isolated_features_list


def get_mean_of(array):
    return np.mean(array)


def get_std_of(array):
    return np.std(array, ddof=1)


# Make the possibility predictions using Gaussian Naive Bayes Model
def pdf(x, mean, std):
    return (1 / (math.sqrt(2 * math.pi * math.pow(std, 2)))) * math.exp(-((math.pow(x - mean, 2)) / (2 * math.pow(std, 2))))


# For all the features of the iris map: get all the values of a feature 
# and add them all into its corresponding list;
#
# Refer to 'isolated_feature' for more information on parameter 'section'
#
def set_iris_isolated_features(iris, section):
    iris['sepal length'] = isolated_feature('sepal_length', section)
    iris['sepal width'] = isolated_feature('sepal_width', section)
    iris['petal length'] = isolated_feature('petal_length', section)
    iris['petal width'] = isolated_feature('petal_width', section)


def calc_iris_mean(iris):
    return [get_mean_of(iris['sepal length']), get_mean_of(iris['sepal width']), get_mean_of(iris['petal length']), get_mean_of(iris['petal width'])]


def calc_iris_std(iris):
    return [get_std_of(iris['sepal length']), get_std_of(iris['sepal width']), get_std_of(iris['petal length']), get_std_of(iris['petal width'])]


def calc_prediction(test, iris):
    prediction = 1.0
    for i in range(0, 4):
        prediction *= pdf(test[i], iris['mean'][i], iris['std'][i])

    return prediction


# -> On call
#
# Perform Gaussian Naive Bayes Classifier based on the number of irises choosen from
# each class for the training_set, and the test array of feature values to test how accurate
# the classifier is.
#
def run_gnb(class_samples_num, test):

    # setup the initial training data
    retrieve_data_from(dataset_file)
    draw_samples_for_training(class_samples_num)
    
    if debug:
        print("Iris dataset size: ", len(iris_dataset))
        print_contents_of(iris_dataset)
    
        print("Training set size: ", len(training_set))
        print_contents_of(training_set)
    
    # create maps to store the iris features
    setosa = {
        'mean' : [],
        'std'  : [],
        'sepal length': [],
        'sepal width' : [],
        'petal length': [],  
        'petal width' : []
    }
    
    
    versicolor = {
        'mean' : [],
        'std'  : [],
        'sepal length': [],
        'sepal width' : [], 
        'petal length': [],  
        'petal width' : []
    }
    
    virginica = {
        'mean' : [],
        'std'  : [],
        'sepal length': [],
        'sepal width' : [],
        'petal length': [],  
        'petal width' : []
    }
    
    # put all the values of each feature of the iris into its corresponding list
    set_iris_isolated_features(setosa, 1)
    set_iris_isolated_features(versicolor, 2)
    set_iris_isolated_features(virginica, 3)
    
    # calculate the mean and standard deviation for prediction
    setosa['mean'] = calc_iris_mean(setosa)
    setosa['std'] = calc_iris_std(setosa)
    versicolor['mean'] = calc_iris_mean(versicolor)
    versicolor['std'] = calc_iris_std(versicolor)
    virginica['mean'] = calc_iris_mean(virginica)
    virginica['std'] = calc_iris_std(virginica)
    
    if debug:
        print("Setosa: ")
        print("mean: ", setosa['mean'])
        print("std: ", setosa['std'])
        
        print("Versicolor: ")
        print("mean: ", versicolor['mean'])
        print("std: ", versicolor['std'])
        
        print("Virginica: ")
        print("mean: ", virginica['mean'])
        print("std: ", virginica['std'])
    
    # calculate the prediction for each iris class
    setosa_prediction = calc_prediction(test, setosa)
    versicolor_prediction = calc_prediction(test, versicolor)
    virginica_prediction = calc_prediction(test, virginica)
    
    print("setosa: {}".format(setosa_prediction))
    print("versicolor: {}".format(versicolor_prediction))
    print("virginica: {}".format(virginica_prediction))

run_gnb(25, test)
