import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.datasets.genre_dataset import GenreDataset
from src.models.artist_classifier import ArtistClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ]
)
genres_folder_path = "../../dataset_files/genres"
genres = [
    genre
    for genre in os.listdir(genres_folder_path)
    if os.path.isdir(os.path.join(genres_folder_path, genre))
]
print(genres)

for genre in genres:
    genre_dataset = GenreDataset(
        csv_file="../../dataset_files/csv/artists.csv",
        img_dir="../../dataset_files/resized",
        transform=transform,
        genre=genre,
        data_type="training",
    )

    from torch.utils.data import random_split

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
    artist_training_loader, artist_validation_loader = get_data_loaders(genre_dataset)

    # Initialize the artist classifier model
    number_of_artists = len(genre_dataset.artist_label_encoder.classes_)
    artist_model = ArtistClassifier(number_of_artists).to(device)

    # Define loss function and optimizer
    artist_loss_fn = (
        torch.nn.CrossEntropyLoss()
    )  # Assuming you're using a single label per image for artists
    artist_optimizer = torch.optim.SGD(
        artist_model.parameters(), lr=0.001, momentum=0.9
    )

    EPOCHS = 4
    # Training loop
    # Training loop
    best_vloss = 1_000_000.0
    best_model_state = None
    for epoch in range(EPOCHS):
        artist_model.train()
        running_loss = 0.0

        for i, data in enumerate(artist_training_loader):
            images, artist_labels = data
            images, artist_labels = images.to(device), artist_labels.to(
                device
            )  # Move to the same device

            artist_optimizer.zero_grad()
            artist_outputs = artist_model(images)

            # Direct conversion in loss function call
            loss = artist_loss_fn(artist_outputs, artist_labels.type(torch.long))
            loss.backward()
            artist_optimizer.step()

            running_loss += loss.item()
            # Additional print statements
            print(f"Batch: {i}, Device: {artist_labels.device}, Loss: {running_loss}")

        # Validation step
        artist_model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, artist_labels in artist_validation_loader:
                images, artist_labels = images.to(device), artist_labels.to(device)
                artist_outputs = artist_model(images)
                loss = artist_loss_fn(artist_outputs, artist_labels.type(torch.long))
                validation_loss += loss.item()

        avg_validation_loss = validation_loss / len(artist_validation_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_validation_loss}")
        if avg_validation_loss < best_vloss:
            best_vloss = avg_validation_loss
            best_model_state = (
                artist_model.state_dict().copy()
            )  # Store the best model state
            print(
                f"Best model updated at epoch {epoch + 1} with validation loss: {avg_validation_loss}"
            )

    if best_model_state is not None:
        model_save_name = (
            f"{genre_dataset.genre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        model_path = f"../../runs/genre_specific/{model_save_name}"
        torch.save(best_model_state, model_path)
        print(f"Saved best model with validation loss: {best_vloss}")
