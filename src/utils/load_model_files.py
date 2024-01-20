import os
from collections import defaultdict

import torch
import torchvision.models as models

from src.models.artist_classifier import ArtistClassifier
from src.models.artist_classifier import *


def get_genre_classifier():

    genre_model_path = "../../runs/genre_models/genre_model1.pth"
    genre_model = modify_resnet_model(models.resnet18(pretrained=True), num_classes=10)

    genre_model.load_state_dict(torch.load(genre_model_path))
    genre_model.eval()

    return genre_model


def get_artist_classifiers():
    artist_models = defaultdict()
    genre_specific_models_path = "../../runs/genre_specific"
    genre_specific_model_files = os.listdir(genre_specific_models_path)

    for file in genre_specific_model_files:
        genre_model_path = os.path.join(genre_specific_models_path, file)
        genre_key_arr = file.split(".")
        genre_key = genre_key_arr[0]

        artist_count = 36
        genre_model = modify_resnet_model(models.resnet18(pretrained=True), num_classes=36)

        genre_model.load_state_dict(torch.load(genre_model_path))

        artist_models[genre_key] = genre_model
        genre_model.eval()
    return artist_models
