import os
import random
import shutil


def exclude_test_data(dataset_path):

    test_data_path = os.path.join(dataset_path, "test")

    try:
        os.makedirs(test_data_path)
    except FileExistsError:
        pass
        #raise FileExistsError('Test data is arleady creatd')

    files = [f for f in os.listdir(dataset_path) if
             os.path.isfile(os.path.join(dataset_path, f))]

    random.shuffle(files)
    test_dataset_size = int(0.3 * len(files))

    test_files = files[:test_dataset_size]

    for file in test_files:
        source_path = os.path.join(dataset_path, file)
        destination_path = os.path.join(test_data_path, file)

        shutil.copyfile(source_path, destination_path)
        os.remove(file)

    # Coping and deleting part
    # Inside evry genree there will be test file
    # bcz wehn we will test it we will use the data from test folder and we will
#     fed the data to the first classifer and based on what it gere it produce we
# will select our second model to predict the artist name
# bcz we are training classider on each genre so after the first model makes a new 


if __name__ == "__main__":
    whole_data_set_path = "../../dataset_files/resized"
    #exclude_test_data(whole_data_set_path)

    genres_datasets_path = os.path.join(whole_data_set_path, "genres")
    genres_dirs = files = [f for f in os.listdir(genres_datasets_path) if
                           os.path.isdir(os.path.join(genres_datasets_path, f)) and f != "test"]

    for genres_dir in genres_dirs:
        print(genres_dir)
        #exclude_test_data(genres_dir)


