import os
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image

# Dataset class
import os

import numpy as np
import torch
from PIL import Image


from torch.utils.data import Dataset


class GenreDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        genre: str,
        transform=None,
    ):
        super().__init__(csv_file, img_dir, transform)

        self.genre = genre
        self.artists_names = []
        self.genre_img_path = os.path.join(img_dir, "genres", self.genre)

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


genres_path = r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files'
cv_path =r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\csv\artists.csv'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = GenreDataset(genres_path,'cubism',transform)


def get_data_loaders(dataset, batch_size=32):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


# Create DataLoaders using the get_data_loaders function
artist_training_loader, artist_validation_loader = get_data_loaders(dataset)

for image,artist in artist_training_loader:
     print(image.shape,artist.shape)
     print(dataset.decode_artist_label(artist))
