import os
import random
import shutil


def split_single_folder_data(dataset_path):
    test_data_path = os.path.join(dataset_path, "test")
    training_data_path = os.path.join(dataset_path, "training")

    os.makedirs(test_data_path)
    os.makedirs(training_data_path)

    files = [
        f
        for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f))
    ]

    random.shuffle(files)
    test_dataset_size = int(0.3 * len(files))

    test_files = files[:test_dataset_size]
    training_files = files[test_dataset_size:]

    for file in test_files:
        source_path = os.path.join(dataset_path, file)
        destination_path = os.path.join(test_data_path, file)

        shutil.copyfile(source_path, destination_path)
        os.remove(source_path)

    for file in training_files:
        source_path = os.path.join(dataset_path, file)
        destination_path = os.path.join(training_data_path, file)

        shutil.copyfile(source_path, destination_path)
        os.remove(source_path)


if __name__ == "__main__":
    whole_data_set_path = "../../dataset_files/resized"
    split_single_folder_data(whole_data_set_path)

    genres_datasets_path = os.path.join(whole_data_set_path, "genres")
    genres_dirs = [
        f
        for f in os.listdir(genres_datasets_path)
        if os.path.isdir(os.path.join(genres_datasets_path, f)) and f != "test"
    ]

    for genres_dir in genres_dirs:
        genres_dir = os.path.join(genres_datasets_path, genres_dir)
        split_single_folder_data(genres_dir)
