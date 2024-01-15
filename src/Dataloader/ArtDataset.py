import os
import re

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


class ArtDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()

        self.artist_to_data = {}
        self.encoding_info = {}
        self.genres_labels = []
        self.artists_names = []

        self.convert_names()
        self.set_artist_names()
        self.set_genre_labels()

        self.fit_label_encoder()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]
        artist_name = os.path.splitext(img_name)[0]
        artist_name = "_".join(artist_name.split("_")[:-1])

        # Remove trailing number and extension
        image = Image.open(os.path.join(self.img_dir, img_name))
        artist_data = self.artist_to_data.get(artist_name)

        genres = artist_data["genre"].split(",")
        genre_trans = self.label_encoder.transform(genres)
        artist_name_transform = self.label_encoder.transform([artist_name])

        if self.transform:
            image = self.transform(image)

        if artist_data is None:
            raise ValueError(f"No data found for artist: {artist_name}")

        return image, genre_trans, artist_name_transform

    def set_genre_labels(self):
        artists_df = self.data
        unique_genres = set()
        for genres in artists_df["genre"]:
            for genre in genres.split(","):
                unique_genres.add(genre.strip())
        self.genres_labels = sorted(list(unique_genres))

    def set_artist_names(self):
        artists_df = self.data
        unique_artists = set(artists_df["name"])

        for name in unique_artists:
            concatenated_name = name.replace(" ", "_")
            self.artists_names.append(concatenated_name)

    def fit_label_encoder(self):
        self.label_encoder.fit(self.genres_labels + self.artists_names)

    def convert_names(self):
        for _, row in self.data.iterrows():
            processed_name = convert_name(row["name"])
            self.artist_to_data[processed_name] = row

    def encoded_label_to_string(self, encoded_label):
        if len(encoded_label.shape) == 2:
            # Nested list comprehension for 2D tensors
            return [
                [
                    self.label_encoder.inverse_transform([label.item()])[0]
                    for label in row
                ]
                for row in encoded_label
            ]
        else:
            # Handling for 1D tensors
            return [
                self.label_encoder.inverse_transform([label.item()])[0]
                for label in encoded_label
            ]
