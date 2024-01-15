import os
import re

import numpy as np
import pandas as pd
import unidecode
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


def convert_name(artist_name):
    if artist_name == "Albrecht Dürer":
        return "Albrecht_Dürer"
    if artist_name == "Pierre-Auguste Renoir":
        return "Pierre-Auguste_Renoir"
    if artist_name == "Henri de Toulouse-Lautrec":
        return "Henri_de_Toulouse-Lautrec"

    # Convert to ASCII, replace spaces with underscores, and remove special characters
    name = unidecode.unidecode(artist_name)
    name = name.replace(" ", "_")
    name = re.sub(r"\W+", "", name)  # Removes any remaining non-alphanumeric characters

    return name
def mapping_genre_to_class(path, column):
    artists_df = pd.read_csv(path)
    unique_genres = set()
    for genres in artists_df[column]:
        for genre in genres.split(','):
            unique_genres.add(genre.strip())
    unique_genres_list = sorted(list(unique_genres))
    genre_to_label = {genre: i for i, genre in enumerate(unique_genres_list)}
    return genre_to_label

def mapping_artist_to_class(path, column):
    artists_df = pd.read_csv(path)
    unique_artists = set(artists_df[column])
    artist_to_label = {artist: i for i, artist in enumerate(sorted(unique_artists))}
    return artist_to_label

class ArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.artist_to_data = {}
        self.encoding_info = {}

        # Use the mapping_genre_to_class function
        self.genre_to_label = mapping_genre_to_class(csv_file, 'genre')
        self.artist_to_label = mapping_artist_to_class(csv_file, 'name')
        self.convert_names()
        self.fit_label_encoder()


    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]
        artist_name = os.path.splitext(img_name)[0]
        artist_name = "_".join(artist_name.split("_")[:-1])
        artist_data = self.artist_to_data.get(artist_name)
        print(artist_name, "-" * 100, '\n', self.artist_to_data)
        #
        # import pdb;
        # pdb.set_trace()
        if artist_data is None:
            raise ValueError(f"No data found for artist: {artist_name}")

        # Convert genre and artist to numerical labels
        genres = artist_data["genre"].split(',')
        genre_labels = [self.genre_to_label[genre.strip()] for genre in genres]
        artist_label = self.artist_to_label[artist_name]

        # Load and transform image
        image = Image.open(os.path.join(self.img_dir, img_name))
        if self.transform:
            image = self.transform(image)

        return image, genre_labels, artist_label
    def fit_label_encoder(self):
        labels = []
        artists = []

        for artist_name in self.artist_to_data:
            genre = self.artist_to_data[artist_name]["genre"]

            labels.append(genre)
            artists.append(artist_name)

        labels = list(set(labels))
        artists = list(set(artists))

        self.label_encoder.fit(labels + artists)

    def convert_names(self):
        for _, row in self.data.iterrows():
            processed_name = convert_name(row["name"])
            self.artist_to_data[processed_name] = row

    def label_to_string(self, encoded_label):
        # Ensure that encoded_label is a 1D array or a single integer
        if isinstance(encoded_label, int):
            encoded_label = [encoded_label]
        elif isinstance(encoded_label, np.ndarray) and encoded_label.ndim > 1:
            # Flatten the array if it's multi-dimensional
            encoded_label = encoded_label.flatten()

        return self.label_encoder.inverse_transform(encoded_label)[0]
