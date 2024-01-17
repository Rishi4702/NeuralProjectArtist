import os

import numpy as np
import torch
from PIL import Image

from src.datasets.art_dataset import ArtDataset


class GenreDataset(ArtDataset):
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        genre: str,
        data_type="training",
        transform=None,
    ):
        super().__init__(csv_file, img_dir, data_type, transform)

        self.genre = genre
        self.artists_names = []
        self.genre_img_path = os.path.join(img_dir, "genres", self.genre, data_type)

        self.set_authors()
        self.prepare_csv_data_for_single_genre()

    def __getitem__(self, idx):
        img_name = os.listdir(self.genre_img_path)[idx]

        artist_name = os.path.splitext(img_name)[0]
        artist_name = "_".join(artist_name.split("_")[:-1])

        image = Image.open(os.path.join(self.genre_img_path, img_name))
        artist_data = self.artist_to_data.get(artist_name)

        if artist_data is None:
            raise ValueError(f"No data found for artist: {artist_name}")

        if self.transform:
            image = self.transform(image)

        artist = self.artist_label_encoder.transform([artist_name])[0]
        return image, artist

    def __len__(self):
        return len(os.listdir(self.genre_img_path))

    def set_authors(self):
        for index, row in self.data.iterrows():
            genres = row["genre"].split(",")

            for genre in genres:
                genre = genre.strip()

                if genre == self.genre:
                    self.artists_names.append(row["name"])

    def prepare_csv_data_for_single_genre(self):
        for i, row in enumerate(self.data.iterrows()):
            name_of_artist = row[1]["name"]

            if name_of_artist not in self.artists_names:
                self.data.drop(i, axis=0, inplace=True)

    def decode_artist_label(self, encoded_label):
        # Decodes a single label or a list/array of labels
        if isinstance(encoded_label, (list, torch.Tensor, np.ndarray)):
            return self.artist_label_encoder.inverse_transform(encoded_label)
        else:
            return self.artist_label_encoder.inverse_transform([encoded_label])[0]
