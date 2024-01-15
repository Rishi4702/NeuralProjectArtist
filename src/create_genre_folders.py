import os
import re

import pandas as pd
from collections import defaultdict
import shutil
import glob

# Load data from the CSV file
df = pd.read_csv("../Dataset/artists.csv")

# Initialize an empty genre-based dictionary
names_dict = defaultdict(list)

# Iterate through the DataFrame rows
for index, row in df.iterrows():
    genres = row['genre'].split(',')
    name = row['name']
    name = name.replace(' ', '_')

    for genre in genres:
        genre = genre.strip()  # Remove leading/trailing spaces
        names_dict[name].append(genre)

orginal_path = '../Dataset/resized'
path_to_copy = '../Dataset/genres'

img_list = os.listdir(orginal_path)

for img_path in img_list:
    source_path = os.path.join(orginal_path, img_path)

    artist_arr = img_path.split('_')[:-1]
    artist_name = '_'.join(artist_arr)

    for genre in names_dict[artist_name]:
        destination_path = os.path.join(path_to_copy, genre)

        try:
            os.makedirs(destination_path)
        except FileExistsError:
            continue

        destination_path = os.path.join(destination_path, img_path)
        shutil.copyfile(source_path, destination_path)
