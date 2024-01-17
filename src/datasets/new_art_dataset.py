import os
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class ArtDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file)
        self.genre_label_encoder = LabelEncoder()
        self.genres_labels = self.set_genre_labels()
        self.fit_label_encoders()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]
        image = Image.open(os.path.join(self.img_dir, img_name))

        # Extract artist's name from the image filename
        artist_name = "_".join(img_name.split("_")[:-1]).replace('_', ' ')

        # Find the corresponding genre for the artist in the CSV
        genre = self.data.loc[self.data['name'] == artist_name, 'genre'].values[0]
        genre_encoded = self.genre_label_encoder.transform([genre])[0]

        genre_tensor = torch.tensor(genre_encoded, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, genre_tensor

    def set_genre_labels(self):
        unique_genres = set(self.data['genre'])
        print(len(unique_genres))
        return sorted(list(unique_genres))

    def num_gen(self):
        unique_genres = set(self.data['genre'])
        return len(unique_genres)
    def fit_label_encoders(self):
        self.genre_label_encoder.fit(self.genres_labels)

    def decode_label_to_string(self, encoded_label):
        return self.genre_label_encoder.inverse_transform([encoded_label])[0]


# # Example usage:
# dataset = ArtDataset(csv_file=r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\csv\artist.csv', img_dir=r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\resized')
#
# for i in range(len(dataset)):
#     image, genre_encoded = dataset[i]
#     genre_decoded = dataset.decode_label_to_string(genre_encoded)
#
#     print(f"Encoded Label: {genre_encoded}, Decoded Label: {genre_decoded}")
