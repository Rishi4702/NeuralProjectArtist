import os
from collections import defaultdict

import torch
import torchvision.models as models

from src.models.artist_classifier import ArtistClassifier
from src.models.new_genre_classifier import *


def get_genre_classifier():
    """Load fitted GenreClassifier"""

    genre_model_path = "../../runs/genre_models/genre_model1.pth"
    # genre_model_file = os.listdir(genre_model_path)[0]
    # genre_model_file_path = os.path.join(genre_model_path, genre_model_file)

    # Create an instance of the model (update with appropriate arguments if needed)
    genre_count = 24
    genre_model = modify_resnet_model(models.resnet18(pretrained=True), num_classes=10)

    genre_model.load_state_dict(torch.load(genre_model_path))
    genre_model.eval()

    return genre_model


def get_artist_classifiers():
    """Load fitted ArtistClassifiers"""
    artist_models = defaultdict()
    genre_specific_models_path = "../../runs/genre_specific"
    genre_specific_model_files = os.listdir(genre_specific_models_path)

    for file in genre_specific_model_files:
        genre_model_path = os.path.join(genre_specific_models_path, file)
        genre_key_arr = file.split(".")
        genre_key = genre_key_arr[0]

        artist_count = 36
        artist_model = ArtistClassifier(artist_count)
        artist_model.load_state_dict(torch.load(genre_model_path))

        artist_models[genre_key] = artist_model
        artist_model.eval()
    return artist_models
