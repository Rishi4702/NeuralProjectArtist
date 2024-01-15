import os
import re

import pandas as pd
import unidecode
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


def convert_name(artist_name):
    if artist_name == "Albrecht Dürer":
        return "Albrecht_Duâ Ãªrer"
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
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.artist_to_data = {}
        self.encoding_info = {}

        self.convert_names()
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

        genre = artist_data["genre"]
        genre_trans = self.label_encoder.transform([genre])
        artist_name_transform = self.label_encoder.transform([artist_name])

        if self.transform:
            image = self.transform(image)

        if artist_data is None:
            raise ValueError(f"No data found for artist: {artist_name}")

        return image.float(), genre_trans, artist_name_transform

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
        return self.label_encoder.inverse_transform([encoded_label])[0]
