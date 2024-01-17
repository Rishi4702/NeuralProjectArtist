import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.datasets.art_dataset import ArtDataset
from src.utils.load_model_files import (get_artist_classifiers,
                                        get_genre_classifier)


def decode_highest_probability_genre(pred_genre, genre_label_encoder):
    # Apply softmax to get probabilities
    probabilities = F.softmax(pred_genre, dim=1)

    # Find the index of the highest probability
    highest_prob_index = torch.argmax(probabilities, dim=1)

    # Ensure the result is a 1D array
    highest_prob_index_1d = highest_prob_index.cpu().numpy()
    if highest_prob_index_1d.ndim == 0:
        highest_prob_index_1d = np.array([highest_prob_index_1d])

    # Decode the index to the genre name
    highest_prob_genre = genre_label_encoder.inverse_transform(highest_prob_index_1d)

    print(highest_prob_genre)
    return highest_prob_genre



def get_top_prediction_with_probability(pred_artist):
    predicted_probabilities = torch.nn.functional.softmax(pred_artist, dim=1)
    top_probabilities, top_indices = torch.topk(predicted_probabilities, k=1)

    top_class = top_indices[0].item()
    top_probability = top_probabilities[0].item()

    return top_class, top_probability


def genres_tensor_to_label(genre_tensor):
    positions = torch.nonzero(genre_tensor == 1)
    print(positions)


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ]
)

dataset = ArtDataset(
    csv_file="../../dataset_files/csv/artists.csv",
    img_dir="../../dataset_files/resized",
    transform=transform,
    data_type="training",
)
artist_models = get_artist_classifiers()
genre_model = get_genre_classifier()

validation_loader = DataLoader(dataset, batch_size=5)


correct_predictions = 0
data_count = 0

with torch.no_grad():
    for data in validation_loader:
        data_count += 1
        image, genre_tensor, artist_encoded = data
        print(dataset.decode_label_to_string(genre_tensor))
        pred_genre = genre_model(image)

        decoded_labels = decode_highest_probability_genre(
            pred_genre, dataset.genre_label_encoder
        )

        # final_prediction_probability = 0
        # final_artist_prediction = ""
        #
        # # Predicting artist_names with specific aritst classifiers
        # for label in decoded_labels[0]:
        #     artist_classifier = artist_models[label]
        #
        #     predicted_artist = artist_classifier(image)
        #     predicted_artist, probability = get_top_prediction_with_probability(
        #         predicted_artist
        #     )
        #
        #     if probability > final_prediction_probability:
        #         final_prediction_probability = predicted_artist
        #
        # print(predicted_artist)
