import os
import re

import pandas as pd
import torch
import unidecode
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


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
    def __init__(
        self, csv_file: str, img_dir: str, data_type="training", transform=None
    ):
        self.img_dir = os.path.join(img_dir, data_type)
        self.transform = transform

        self.data = pd.read_csv(csv_file)
        self.genre_label_encoder = LabelEncoder()
        self.artist_label_encoder = LabelEncoder()

        self.artist_to_data = {}
        self.genres_labels = []
        self.artists_names = []

        self.convert_names()
        self.set_artist_names()
        self.set_genre_labels()

        self.fit_label_encoders()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]

        artist_name = os.path.splitext(img_name)[0]
        artist_name = "_".join(artist_name.split("_")[:-1])

        image = Image.open(os.path.join(self.img_dir, img_name))
        artist_data = self.artist_to_data.get(artist_name)

        if artist_data is None:
            raise ValueError(f"No data found for artist: {artist_name}")

        genre = artist_data["genre"]
        genre = genre.lower()
        genre = genre.replace(" ", "_")
        genre_encoded = self.genre_label_encoder.transform([genre])[0]
        genre_tensor = torch.tensor(genre_encoded, dtype=torch.long)
        artist_encoded = self.artist_label_encoder.transform([artist_name])[0]

        if self.transform:
            image = self.transform(image)

        return image, genre_tensor, artist_encoded

    def set_genre_labels(self):
        artists_df = self.data
        unique_genres = set()

        for genres in artists_df["genre"]:
            for genre in genres.split(","):
                genre = self.convert_genre(genre)
                unique_genres.add(genre)

        self.genres_labels = sorted(list(unique_genres))

    def set_artist_names(self):
        artists_df = self.data
        unique_artists = set(artists_df["name"])

        for name in unique_artists:
            concatenated_name = name.replace(" ", "_")
            self.artists_names.append(concatenated_name)

    def fit_label_encoders(self):
        self.genre_label_encoder.fit(self.genres_labels)
        self.artist_label_encoder.fit(self.artists_names)

    def num_gen(self):
        unique_genres = set(self.data["genre"])
        return len(unique_genres)

    def num_art(self):
        uniq_artist = set(self.data["name"])
        return len(uniq_artist)

    def convert_names(self):
        for _, row in self.data.iterrows():
            processed_name = convert_name(row["name"])
            self.artist_to_data[processed_name] = row

    def decode_genre_label_to_string(self, encoded_label):
        return self.genre_label_encoder.inverse_transform([encoded_label])

    def decode_artist_label_to_string(self, encoded_label):
        return self.artist_label_encoder.inverse_transform([encoded_label])

    def convert_genre(self, genre):
        genre = genre.lower()
        genre = genre.replace(" ", "_")

        return genre

    def decode_indices_to_genres(self, indices):
        """Convert genre indices to genre strings."""
        genre_labels = []
        for index in indices:
            genre_labels.append(self.genres_labels[index])
        return genre_labels
