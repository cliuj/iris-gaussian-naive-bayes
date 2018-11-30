import numpy as np
import urllib.request
from random import shuffle
import math

debug = True
has_dataset = True
website_hosting_dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset_file = "dataset.txt"
dataset = []

def print_contents_of(array):
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
if (not has_dataset):
    dataset = retrieve_datatset_from(website_hosting_dataset)
    write_dataset_to(dataset_file, dataset)
    if(debug):
        for data in dataset:
            print("{}".format(data))


# Iris object to store the values
class Iris:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, iris_class):
        self.sepal_length = float(sepal_length)
        self.sepal_width = float(sepal_width)
        self.petal_length = float(petal_length)
        self.petal_width = float(petal_width)
        self.iris_class = iris_class

    def to_string(self):
        print("sepal_length: {}, sepal_width: {}, petal_length: {}, petal_width: {}, iris_class: {}"
            .format(self.sepal_length, self.sepal_width, self.petal_length, self.petal_width, self.iris_class))


# parse the data
def parse_file(to_parse_file):
    return [line.rstrip() for line in open(to_parse_file)]

def retrieve_data_from(dataset_file):
    parsed_file = parse_file(dataset_file)
    
    
    # for every iris data in dataset file
    # parse the data and pass it into a new
    # iris object which is then stored in a global
    # list of irises called "iris_dataset"
    for iris in parsed_file:
        iris_features = []
        iris_features = iris.split(',')

        sepal_length = iris_features[0]
        sepal_width = iris_features[1]
        petal_length = iris_features[2]
        petal_width = iris_features[3]
        iris_class = iris_features[4]

        iris_dataset.append(Iris(sepal_length, sepal_width, petal_length, petal_width, iris_class))

def draw_samples_for_training(class_sample_num):
    for iris_class in range(0, 3): 
        t = 0
        for i in range(iris_class * 50, (iris_class + 1) * 50):
            if t >= class_sample_num:
                print(t)
                break
            training_set.append(iris_dataset[t + (iris_class * 50) ])
            t += 1

    print("iris class: ", iris_class)




isolated_sepal_length = []
isolated_sepal_width = []
isolated_petal_length = [] 
isolated_petal_width = []

def isolate_features():
    for iris in iris_dataset:
        isolated_sepal_length.append(iris.sepal_length)   
        isolated_sepal_width.append(iris.sepal_width)
        isolated_petal_length.append(iris.petal_length)
        isolated_petal_width.append(iris.petal_width)

def sort_iris():
    pass



def get_mean_of(array):
    return np.mean(array)

def get_std_of(array):
    return np.std(array, ddof=1)


iris_dataset = []
training_set = []
retrieve_data_from(dataset_file)
draw_samples_for_training(25)


if debug:
    print("Iris dataset size: ", len(iris_dataset))
    print_contents_of(iris_dataset)

    print("Training set size: ", len(training_set))
    print_contents_of(training_set)



isolate_features()
print("Sepal length mean: ", get_mean_of(isolated_sepal_length))
print("Sepal width mean: ", get_mean_of(isolated_sepal_width))
print("Petal length mean: ", get_mean_of(isolated_petal_length))
print("Petal width mean: ", get_mean_of(isolated_petal_width))


print("Sepal length std: ", get_std_of(isolated_sepal_length))
print("Sepal width std: ", get_std_of(isolated_sepal_width))
print("Petal length std: ", get_std_of(isolated_petal_length))
print("Petal width std: ", get_std_of(isolated_petal_width))



setosa = {
    'mean' : [],
    'std': []
}


versicolor = {
    'mean' : [],
    'std': []
}

virginica = {
    'mean' : [],
    'std': []
}


# make the predictions using Gaussian Naive Bayes Model

def pdf(x, mean, std):
    return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((x - math.pow(mean, 2)) / (2 * math.pow(std, 2))))






