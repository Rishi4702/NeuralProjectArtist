import os
import torchvision.models as models
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src.datasets.genre_dataset import GenreDataset
from src.models.artist_classifier import *

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
    # Initialize lists for storing metrics
    valid_losses = []
    accuracies = []
    genre_dataset = GenreDataset(
        csv_file="../../dataset_files/csv/artists.csv",
        img_dir="",
        transform=transform,
        genre=genre,
        data_type="training",
    )
    genre_dataset.genre_img_path = fr'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\genres\{genre}'

    from torch.utils.data import random_split

    def get_data_loaders(dataset, batch_size=32):
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    artist_training_loader, artist_validation_loader = get_data_loaders(genre_dataset)
    number_of_artists = len(genre_dataset.artist_label_encoder.classes_)

    model = models.resnet18(pretrained=True)
    model = modify_resnet_model(model, number_of_artists)
    artist_model = model.to(device)

    artist_loss_fn = nn.CrossEntropyLoss()
    artist_optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(artist_optimizer, step_size=3, gamma=0.1)

    EPOCHS = 7

    best_vloss = 1_000_000.0
    best_model_state = None
    for epoch in range(EPOCHS):
        artist_model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, data in enumerate(artist_training_loader):
            images, artist_labels = data
            images, artist_labels = images.to(device), artist_labels.to(
                device
            )

            artist_optimizer.zero_grad()
            artist_outputs = artist_model(images)

            loss = artist_loss_fn(artist_outputs, artist_labels.type(torch.long))
            loss.backward()
            artist_optimizer.step()

            running_loss += loss.item()
            _, predicted_indices = torch.max(artist_outputs, 1)
            correct_predictions += (predicted_indices == artist_labels).sum().item()
            total_predictions += artist_labels.size(0)
            print(f"Batch: {i}, Device: {artist_labels.device}, Loss: {running_loss}")

        avg_train_loss = running_loss / len(artist_training_loader)
        train_accuracy = correct_predictions / total_predictions

        artist_model.eval()
        validation_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for images, artist_labels in artist_validation_loader:
                images, artist_labels = images.to(device), artist_labels.to(device)
                artist_outputs = artist_model(images)
                loss = artist_loss_fn(artist_outputs, artist_labels.type(torch.long))
                validation_loss += loss.item()
                _, predicted_indices = torch.max(artist_outputs, 1)
                correct_predictions += (predicted_indices == artist_labels).sum().item()
                total_predictions+=len(artist_labels)

        avg_validation_loss = validation_loss / len(artist_validation_loader)
        valid_losses.append(avg_validation_loss)

        if total_predictions ==0 :
            accuracies.append(0)
        else:
            validation_accuracy = correct_predictions / total_predictions
            accuracies.append(validation_accuracy)

    if avg_validation_loss < best_vloss:
            best_vloss = avg_validation_loss
            best_model_state = (
                artist_model.state_dict().copy()
            )
            print(
                f"Best model updated at epoch {epoch + 1} with validation loss: {avg_validation_loss}"
            )


    # Plot training and validation loss
    # Plotting for each genre
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 6))

    # Plot validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, valid_losses, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracies, label='Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot for the current genre
    plt_save_path = os.path.join("../../runs/genre_specific", f"{genre}_validation_plot.png")
    plt.savefig(plt_save_path)
    plt.close()

if best_model_state is not None:
        model_save_name = f"{genre_dataset.genre}.pt"
        model_path = f"../../runs/genre_specific/{model_save_name}"
        torch.save(best_model_state, model_path)
        print(f"Saved best model with validation loss: {best_vloss}")
