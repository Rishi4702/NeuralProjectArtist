import os
import shutil
from collections import defaultdict

import pandas as pd

df = pd.read_csv("../../dataset_files/csv/artists.csv")
names_dict = defaultdict(list)

original_path = "../../dataset_files/resized"
path_to_copy = "../../dataset_files/genres"

if __name__ == "__main__":
    try:
        os.makedirs(path_to_copy)
    except FileExistsError:
        pass

    for index, row in df.iterrows():
        genres = row["genre"].split(",")
        name = row["name"]
        name = name.replace(" ", "_")

        for genre in genres:
            genre = genre.strip()  # Remove leading/trailing spaces
            names_dict[name].append(genre)

    img_list = os.listdir(original_path)

    for img_path in img_list:
        source_path = os.path.join(original_path, img_path)

        artist_arr = img_path.split("_")[:-1]
        artist_name = "_".join(artist_arr)

        for genre in names_dict[artist_name]:
            genre = genre.lower()
            genre = genre.replace(" ", "_")
            destination_path = os.path.join(path_to_copy, genre)

            try:
                os.makedirs(destination_path)
            except FileExistsError:
                pass

            destination_path = os.path.join(destination_path, img_path)
            shutil.copyfile(source_path, destination_path)
