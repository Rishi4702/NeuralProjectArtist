import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.datasets.art_dataset import ArtDataset
from src.utils.load_model_files import (get_artist_classifiers,
                                        get_genre_classifier)


def decode_highest_probability_genre(pred_genre, genre_label_encoder):

    probabilities = F.softmax(pred_genre, dim=1)


    highest_prob_index = torch.argmax(probabilities, dim=1)


    highest_prob_index_1d = highest_prob_index.cpu().numpy()
    if highest_prob_index_1d.ndim == 0:
        highest_prob_index_1d = np.array([highest_prob_index_1d])


    highest_prob_genre = genre_label_encoder.inverse_transform(highest_prob_index_1d)

    return highest_prob_genre[0]


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

validation_loader = DataLoader(dataset, batch_size=1)


correct_artist_predictions = 0
correct_genre_predictions = 0
data_count = 0

with torch.no_grad():
    for data in validation_loader:
        data_count += 1

        image, genre_tensor, artist = data
        decoded_original_genre = dataset.decode_genre_label_to_string(
            genre_tensor.item()
        )
        decoded_original_artist = dataset.decode_artist_label_to_string(artist.item())

        pred_genre = genre_model(image)
        decoded_predicted_genre = decode_highest_probability_genre(
            pred_genre, dataset.genre_label_encoder
        )

        if decoded_original_genre == decoded_predicted_genre:
            correct_genre_predictions += 1

        artist_classifier = artist_models[decoded_predicted_genre]
        predicted_artist = artist_classifier(image)
        predicted_artist, _ = get_top_prediction_with_probability(predicted_artist)
        decoded_predicted_artist = dataset.decode_artist_label_to_string(
            predicted_artist
        )

        if decoded_predicted_artist == decoded_original_artist:
            correct_artist_predictions += 1

accuracy_of_correct_genre_predictions = correct_genre_predictions / data_count
accuracy_of_hierarchical_model = correct_artist_predictions / data_count

print(
    f"accuracy hierarchical model :{accuracy_of_hierarchical_model} accuracy genre: {accuracy_of_correct_genre_predictions}"
)
