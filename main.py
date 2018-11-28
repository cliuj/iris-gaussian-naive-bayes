import urllib.request

debug = True
has_dataset = True
website_hosting_dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset_file = "dataset.txt"
dataset = []


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
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.flower_class = flower_class

    def to_String(self):
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


flowers_dataset = []
retrieve_data_from(dataset_file)

if debug:
    for flower in flowers_dataset:
        flower.to_String()













# perfom naive bayes training
