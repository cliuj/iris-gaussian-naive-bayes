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

        self.features = {
            'sepal_length' : float(sepal_length),
            'sepal_width' : float(sepal_width),
            'petal_length' : float(petal_length),
            'petal_width' : float(petal_width),
            'iris_class' : iris_class
        }
        #self.sepal_length = float(sepal_length)
        #self.sepal_width = float(sepal_width)
        #self.petal_length = float(petal_length)
        #self.petal_width = float(petal_width)
        #self.iris_class = iris_class

    def to_string(self):
        print("sepal_length: {}, sepal_width: {}, petal_length: {}, petal_width: {}, iris_class: {}"
            .format(self.features['sepal_length'],
                    self.features['sepal_width'],
                    self.features['petal_length'],
                    self.features['petal_width'],
                    self.features['iris_class']))



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
        isolated_sepal_length.append(iris.features['sepal_length'])   
        isolated_sepal_width.append(iris.features['sepal_width'])
        isolated_petal_length.append(iris.features['petal_length'])
        isolated_petal_width.append(iris.features['petal_width'])

def isolated_feature(feature, start, end):
    isolated_features_list = []
    for i in range(start, end):
        isolated_features_list.append(training_set[i].features[feature])
    return isolated_features_list


def sort_iris():
    for i in range(0, 25):
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
    return (1 / (math.sqrt(2 * math.pi * math.pow(std, 2)))) * math.exp(-((math.pow(x - mean, 2)) / (2 * math.pow(std, 2))))



setosa_sepal_length = isolated_feature('sepal_length', 0, 25)
setosa_sepal_width = isolated_feature('sepal_width', 0, 25)
setosa_petal_length = isolated_feature('petal_length', 0, 25)
setosa_petal_width = isolated_feature('petal_width', 0, 25)

versicolor_sepal_length = isolated_feature('sepal_length', 25, 50)
versicolor_sepal_width = isolated_feature('sepal_width', 25, 50)
versicolor_petal_length = isolated_feature('petal_length', 25, 50)
versicolor_petal_width = isolated_feature('petal_width', 25, 50)


virginica_sepal_length = isolated_feature('sepal_length', 50, 75)
virginica_sepal_width = isolated_feature('sepal_width', 50, 75)
virginica_petal_length = isolated_feature('petal_length', 50, 75)
virginica_petal_width = isolated_feature('petal_width', 50, 75)




setosa['mean'] = [get_mean_of(setosa_sepal_length), get_mean_of(setosa_sepal_width), get_mean_of(setosa_petal_length), get_mean_of(setosa_petal_width)]
setosa['std'] = [get_std_of(setosa_sepal_length), get_std_of(setosa_sepal_width), get_std_of(setosa_petal_length), get_std_of(setosa_petal_width)]



versicolor['mean'] = [get_mean_of(versicolor_sepal_length), get_mean_of(versicolor_sepal_width), get_mean_of(versicolor_petal_length), get_mean_of(versicolor_petal_width)]

versicolor['std'] = [get_std_of(versicolor_sepal_length), get_std_of(versicolor_sepal_width), get_std_of(versicolor_petal_length), get_std_of(versicolor_petal_width)]



virginica['mean'] = [get_mean_of(virginica_sepal_length), get_mean_of(virginica_sepal_width), get_mean_of(virginica_petal_length), get_mean_of(virginica_petal_width)]

virginica['std'] = [get_std_of(virginica_sepal_length), get_std_of(virginica_sepal_width), get_std_of(virginica_petal_length), get_std_of(virginica_petal_width)]


print(setosa['mean'])
print(setosa['std'])


print(versicolor['mean'])
print(versicolor['std'])


print(virginica['mean'])
print(virginica['std'])


test = [5.7, 2.8, 4.1, 1.3]



setosa_prediction = pdf(test[0], setosa['mean'][0], setosa['std'][0]) * pdf(test[1], setosa['mean'][1], setosa['std'][1]) * pdf(test[2], setosa['mean'][2], setosa['std'][2]) * pdf(test[3], setosa['mean'][3], setosa['std'][3]) 

versicolor_prediction = pdf(test[0], versicolor['mean'][0], versicolor['std'][0]) * pdf(test[1], versicolor['mean'][1], versicolor['std'][1]) * pdf(test[2], versicolor['mean'][2], versicolor['std'][2]) * pdf(test[3], versicolor['mean'][3], versicolor['std'][3]) 

virginica_prediction = pdf(test[0], virginica['mean'][0], virginica['std'][0]) * pdf(test[1], virginica['mean'][1], virginica['std'][1]) * pdf(test[2], virginica['mean'][2], virginica['std'][2]) * pdf(test[3], virginica['mean'][3], virginica['std'][3]) 



print("setosa: {}".format(setosa_prediction))
print("versicolor: {}".format(versicolor_prediction))
print("virginica: {}".format(virginica_prediction))
