import urllib.request
from random import shuffle

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
        if setosa == 25:
            break
        training_set.append(flowers_dataset[setosa])

    for versicolor in range(50, 100):
        if versicolor == 75:
            break
        training_set.append(flowers_dataset[versicolor])

    for virginica in range(100, len(flowers_dataset)):
        if virginica == 125:
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


sepal_length_mean(setosa, 0, 25)
sepal_width_mean(setosa,  0, 25)
petal_length_mean(setosa, 0, 25)
petal_width_mean(setosa,  0, 25)

sepal_length_mean(versicolor,25, 50)
sepal_width_mean(versicolor, 25, 50)
petal_length_mean(versicolor,25, 50)
petal_width_mean(versicolor, 25, 50)

sepal_length_mean(virginica, 50, 75)
sepal_width_mean(virginica,  50, 75)
petal_length_mean(virginica, 50, 75)
petal_width_mean(virginica,  50, 75)


print("setosa: \t{}".format(setosa['mean']))
print("versicolor: \t{}".format(versicolor['mean']))
print("virginica: \t{}".format(virginica['mean']))
