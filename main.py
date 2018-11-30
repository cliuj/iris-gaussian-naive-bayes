import urllib.request
from random import shuffle
import math


debug = True
has_dataset = True
website_hosting_dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset_file = "dataset.txt"
dataset = []


def print_contents_of(array):
    for data in array:
        data.to_string()   
    print()



# The following both retrieves the UTF-8 encoded data and
# decodes the data into string data.
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

# Set has_dataset to False if dataset.txt is missing or empty
if (not has_dataset):
    dataset = retrieve_dataset_from(website_hosting_dataset)
    write_dataset_to(dataset_file, dataset)
    if (debug):
        for data in dataset:
            print("{}".format(data))



# plant object to store values

class Flower:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, flower_class):
        self.sepal_length = float(sepal_length)
        self.sepal_width  = float(sepal_width)
        self.petal_length = float(petal_length)
        self.petal_width  = float(petal_width)
        self.flower_class = flower_class

    def to_string(self):
        print("sepal_length: {}, sepal_width: {}, petal_length: {}, petal_width: {}, flower_class: {}".format(self.sepal_length, self.sepal_width, self.petal_length, self.petal_width, self.flower_class))


# parse the data 
def parse_file(to_parse_file):
    return [line.rstrip() for line in open(to_parse_file)]

def retrieve_data_from(dataset_file):
    parsed_file = parse_file(dataset_file)

    iris_dataset = []
    
    # for every flower data in the dataset file
    # parse the data and store it into a
    # flower object    
    for flower in parsed_file:
        data_info = []
        #print("{}".format(flower))
        data_info = flower.split(',')
        #print(data_info)
        iris_dataset.append(data_info)

        sepal_length = data_info[0]
        sepal_width = data_info[1]
        petal_length = data_info[2]
        petal_width = data_info[3]
        flower_class = data_info[4]

        flowers_dataset.append(Flower(sepal_length, sepal_width, petal_length, petal_width, flower_class))


def draw_samples_for_training():
    for setosa in range(0, 50):
        if setosa == 50:
            break
        training_set.append(flowers_dataset[setosa])

    for versicolor in range(50, 100):
        if versicolor == 100:
            break
        training_set.append(flowers_dataset[versicolor])

    for virginica in range(100, len(flowers_dataset)):
        if virginica == 150:
            break
        training_set.append(flowers_dataset[virginica])



# get the data from dataset.txt and store the data into
# a list of flowers
flowers_dataset = []
retrieve_data_from(dataset_file)

#if debug:
#    print("Unshuffled flowers dataset:")
#    print_contents_of(flowers_dataset)
#
#shuffle(flowers_dataset)
#if debug:
#    print("Shuffled flowers dataset:")
#    print_contents_of(flowers_dataset)


# setup the training set
training_set = []

draw_samples_for_training()

if debug:
    print("Training set:")
    print_contents_of(training_set)

if debug:
    print("Size of flowers_dataset: {}".format(len(flowers_dataset)))
    print("Size of training_set: {}".format(len(training_set)))



# perfom naive bayes training


# this is used to check if my calculations are correct
# get mean and total of all samples
def mean():
    sepal_length_sigma = 0
    sepal_width_sigma = 0
    petal_length_sigma = 0
    petal_width_sigma = 0
    
    for flower in flowers_dataset:
        sepal_length_sigma += flower.sepal_length
        sepal_width_sigma += flower.sepal_width
        petal_length_sigma += flower.petal_length
        petal_width_sigma += flower.petal_width

    
    sepal_length_mean = sepal_length_sigma / 150
    sepal_width_mean = sepal_width_sigma / 150
    petal_length_mean = petal_length_sigma / 150
    petal_width_mean = petal_width_sigma / 150

    print("sepal lenth: {}, sepal width: {}, petal length: {}, petal width: {}".format(sepal_length_mean, sepal_width_mean, petal_length_mean, petal_width_mean))

    std(sepal_length_mean, sepal_width_mean, petal_length_mean, petal_width_mean)


def std(sepal_length_mean, sepal_width_mean, petal_length_mean, petal_width_mean):
    
    sepal_length_dif_mean_s = 0
    sepal_width_dif_mean_s = 0
    petal_length_dif_mean_s = 0
    petal_width_dif_mean_s = 0
    
    for flower in flowers_dataset:
        sepal_length_dif_mean_s += math.pow(flower.sepal_length - sepal_length_mean, 2)
        sepal_width_dif_mean_s +=  math.pow(flower.sepal_width - sepal_width_mean, 2)
        petal_length_dif_mean_s += math.pow(flower.petal_length - petal_length_mean, 2)
        petal_width_dif_mean_s +=  math.pow(flower.petal_width - petal_width_mean, 2)



    sepal_length_sample_variance = sepal_length_dif_mean_s / (150 - 1) 
    sepal_width_sample_variance =  sepal_width_dif_mean_s  / (150 - 1)   
    petal_length_sample_variance = petal_length_dif_mean_s / (150 - 1)
    petal_width_sample_variance =  petal_width_dif_mean_s  / (150 - 1)


    sepal_length_std = math.sqrt(sepal_length_sample_variance)
    sepal_width_std = math.sqrt(sepal_width_sample_variance)
    petal_length_std =math.sqrt(petal_length_sample_variance)
    petal_width_std = math.sqrt(petal_width_sample_variance)
    print("STD:")
    print("sepal lenth: {}, sepal width: {}, petal length: {}, petal width: {}".format(sepal_length_sample_variance, sepal_width_sample_variance, petal_length_sample_variance, petal_width_sample_variance))



# get the mean of the flower's features
def sepal_length_mean(flower_class, start, end):
    sigma = 0
    mean = 0
    for flower in range(start, end):
        sigma = sigma + training_set[flower].sepal_length
    mean = sigma / (end - start)
    flower_class['mean'].append(mean)

def sepal_width_mean(flower_class, start, end):
    sigma = 0
    mean = 0
    for flower in range(start, end):
        sigma = sigma + training_set[flower].sepal_width
    mean = sigma / (end - start)
    flower_class['mean'].append(mean)

def petal_length_mean(flower_class, start, end):
    sigma = 0
    mean = 0
    for flower in range(start, end):
        sigma = sigma + training_set[flower].petal_length
    mean = sigma / (end - start)
    flower_class['mean'].append(mean)

def petal_width_mean(flower_class, start, end):
    sigma = 0
    mean = 0
    for flower in range(start, end):
        sigma = sigma + training_set[flower].petal_width
    mean = sigma / (end - start)
    flower_class['mean'].append(mean)

# get the standard deviation of the flower's features

def sepal_length_std(sepal_length_mean, start, end):
    
    #(xi - mean)
    checker = 0
    temp = 0
    for flower in range(start, end):
        temp = (training_set[flower].sepal_length - sepal_length_mean)
        checker = temp + checker
    #    print("sepal_length: {} - sepal_length_mean: {} = {}".format(training_set[flower].sepal_length, sepal_length_mean, temp))
        
    print("checker ", checker)


def sepal_width_std(sepal_width_mean, start, end):
    
    #(xi - mean)
    checker = 0
    temp = 0
    sigma_square_dif_mean = 0
    for flower in range(start, end):
        temp = (training_set[flower].sepal_width - sepal_width_mean)
        sigma_square_dif_mean = sigma_square_dif_mean + math.pow(temp, 2)
        checker = temp + checker
    #    print("sepal_length: {} - sepal_length_mean: {} = {}".format(training_set[flower].sepal_length, sepal_length_mean, temp))
        
    sample_variance = sigma_square_dif_mean / ((end - start) - 1)
    sample_standard_deviation = math.sqrt(sample_variance)
    print("standard_deviation: ", sample_standard_deviation)
    #print("checker ", checker)


def petal_length_std(petal_length_mean, start, end):
    
    #(xi - mean)
    checker = 0
    temp = 0
    for flower in range(start, end):
        temp = (training_set[flower].petal_length - petal_length_mean)
        checker = temp + checker
    #    print("sepal_length: {} - sepal_length_mean: {} = {}".format(training_set[flower].sepal_length, sepal_length_mean, temp))
        
    print("checker ", checker)


def petal_width_std(petal_width_mean, start, end):
    
    #(xi - mean)
    checker = 0
    temp = 0
    for flower in range(start, end):
        temp = (training_set[flower].petal_width - petal_width_mean)
        checker = temp + checker
    #    print("sepal_length: {} - sepal_length_mean: {} = {}".format(training_set[flower].sepal_length, sepal_length_mean, temp))
        
    print("checker ", checker)


setosa = {
    'mean' : [],
    'std' : []
}

versicolor = {
    'mean' : [],
    'std' : []
}

virginica = {
    'mean' : [],
    'std' : [],
}

mean()

sepal_length_mean(setosa, 0, 50)
sepal_width_mean(setosa,  0, 50)
petal_length_mean(setosa, 0, 50)
petal_width_mean(setosa,  0, 50)

sepal_length_mean(versicolor,50, 100)
sepal_width_mean(versicolor, 50, 100)
petal_length_mean(versicolor,50, 100)
petal_width_mean(versicolor, 50, 100)

sepal_length_mean(virginica, 100, 150)
sepal_width_mean(virginica,  100, 150)
petal_length_mean(virginica, 100, 150)
petal_width_mean(virginica,  100, 150)


print("setosa: \t{}".format(setosa['mean']))
print("versicolor: \t{}".format(versicolor['mean']))
print("virginica: \t{}".format(virginica['mean']))

sepal_length_std((setosa['mean'])[0], 0, 50)
sepal_width_std(( setosa['mean'])[1], 0, 50)
petal_length_std((setosa['mean'])[2], 0, 50)
petal_width_std(( setosa['mean'])[3], 0, 50)

sepal_length_std((versicolor['mean'])[0], 50, 100)
sepal_width_std(( versicolor['mean'])[1], 50, 100)
petal_length_std((versicolor['mean'])[2], 50, 100)
petal_width_std(( versicolor['mean'])[3], 50, 100)

sepal_length_std((virginica['mean'])[0], 100, 150)
sepal_width_std(( virginica['mean'])[1], 100, 150)
petal_length_std((virginica['mean'])[2], 100, 150)
petal_width_std(( virginica['mean'])[3], 100, 150)
