from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import unidecode
import re

def convert_name(artist_name):
    if artist_name == "Albrecht Dürer":
        return "Albrecht_Duâ Ãªrer"
    if artist_name == "Pierre-Auguste Renoir":
        return "Pierre-Auguste_Renoir"
    if artist_name == "Henri de Toulouse-Lautrec":
        return "Henri_de_Toulouse-Lautrec"

    # Convert to ASCII, replace spaces with underscores, and remove special characters
    name = unidecode.unidecode(artist_name)
    name = name.replace(' ', '_')
    name = re.sub(r'\W+', '', name)  # Removes any remaining non-alphanumeric characters
    return name

class ArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.artist_to_data = {}
        for _, row in self.data.iterrows():
            processed_name = convert_name(row['name'])
            self.artist_to_data[processed_name] = row
    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]
        artist_name = os.path.splitext(img_name)[0]
        artist_name = "_".join(artist_name.split("_")[:-1])
        # Remove trailing number and extension

        image = Image.open(os.path.join(self.img_dir, img_name))
        if self.transform:
            image = self.transform(image)

        artist_data = self.artist_to_data.get(artist_name)

        if artist_data is None:
            raise ValueError(f"No data found for artist: {artist_name}")

        return image, artist_data
