import os
import re

import pandas as pd
import torch
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

        genres = artist_data["genre"].split(",")
        genre_encoded = self.genre_label_encoder.transform(genres)

        # Create a binary tensor for genres
        genre_tensor = torch.zeros(len(self.genres_labels), dtype=torch.float32)
        for encoded_label in genre_encoded:
            genre_tensor[encoded_label] = 1  # Set the corresponding index to 1

        if self.transform:
            image = self.transform(image)

        artist_encoded = self.artist_label_encoder.transform([artist_name])[0]

        return image, genre_tensor, artist_encoded

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

    def fit_label_encoders(self):
        self.genre_label_encoder.fit(self.genres_labels)
        self.artist_label_encoder.fit(self.artists_names)

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


class GenreDataset(ArtDataset):
    def __init__(self, csv_file: str, img_dir: str, genre: str, transform=None):
        super().__init__(csv_file, img_dir, transform)

        self.genre = genre
        self.authors = []

        self.set_authors()
        self.delete_unnecessary_data()

    def __getitem__(self, idx):
        pass

    def set_authors(self):
        for index, row in self.data.iterrows():
            genres = row['genre'].split(',')

            for genre in genres:
                genre = genre.strip()

                if genre == self.genre:
                    self.authors.append(row['name'])

    def delete_unnecessary_data(self):
        pass

dataset = GenreDataset(
    csv_file="../../Dataset/artists.csv",
    img_dir="../../Dataset/resized",
    genre='Mannerism'
)

for data in dataset:
    pass

