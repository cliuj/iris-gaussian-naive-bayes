import urllib.request

debug = False
has_dataset = True
website_hosting_dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"



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
if (!has_dataset):
    dataset = retrieve_dataset_from(website_hosting_dataset)
    write_dataset_to("dataset.txt", dataset)
    if (debug):
        for data in dataset:
            print("{}".format(data))



# parse the data and store them into the appropriate parts





















# perfom naive bayes training
