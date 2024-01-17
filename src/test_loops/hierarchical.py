import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.datasets.art_dataset import ArtDataset
from src.utils.load_model_files import (get_artist_classifiers,
                                        get_genre_classifier)


def decode_top_two_predicted_genres(pred_genre, genre_label_encoder):
    # Apply sigmoid to get probabilities
    probabilities = F.softmax(pred_genre, dim=1)

    # Sort probabilities and take indices of top two
    top_two_indices = probabilities.topk(
        2
    ).indices  # Get indices of top two probabilities
    # Decode each pair of indices
    decoded_labels = []
    for indices in top_two_indices:
        genres = genre_label_encoder.inverse_transform(
            indices.cpu().numpy()
        )  # Convert to genre names
        decoded_labels.append(genres)

    return decoded_labels


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
    csv_file="../../dataset_files/artists.csv",
    img_dir="../../dataset_files/resized",
    transform=transform,
    data_type="test",
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

        pred_genre = genre_model(image)
        decoded_labels = decode_top_two_predicted_genres(
            pred_genre, dataset.genre_label_encoder
        )

        final_prediction_probability = 0
        final_artist_prediction = ""

        # Predicting artist_names with specific aritst classifiers
        for label in decoded_labels[0]:
            artist_classifier = artist_models[label]

            predicted_artist = artist_classifier(image)
            predicted_artist, probability = get_top_prediction_with_probability(
                predicted_artist
            )

            if probability > final_prediction_probability:
                final_prediction_probability = predicted_artist

        print(predicted_artist)
