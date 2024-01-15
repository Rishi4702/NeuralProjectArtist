from torch.utils.data import DataLoader
from src.Dataloader.ArtDataset import ArtDataset
from torchvision import transforms
from torch.utils.data import random_split
import warnings
import pandas as pd
warnings.warn("I am UserWarning", UserWarning)
warnings.warn("I am FutureWarning", FutureWarning)


def get_data_loaders(dataset):

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, test_dataloader

def mapping_genre_to_class(path,column):
    artists_df = pd.read_csv(path)
    # Extract unique genres
    unique_genres = set()
    for genres in artists_df[column]:
        # Split genres by comma and add to the set
        for genre in genres.split(','):
            unique_genres.add(genre.strip())  # Remove leading/trailing whitespace

    # Convert the set to a list and sort it
    unique_genres_list = sorted(list(unique_genres))
    genre_to_label = {genre: i for i, genre in enumerate(unique_genres_list)}
    print(genre_to_label)
    return genre_to_label
