from datetime import datetime

import torch
import torchvision.transforms as transforms

from src.datasets.art_dataset import ArtDataset
from src.models.genre_classifier import GenreClassifier
from src.utils.dataloader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(epoch_index):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        image, genres, _ = data
        image = image.to(device)
        genres = genres.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(image)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, genres)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(256, 256)),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = ArtDataset(
    csv_file="../../dataset_files/csv/artists.csv",
    img_dir="../../dataset_files/resized",
    transform=transform,
    data_type="training",
)

training_loader, validation_loader = get_data_loaders(dataset, 0.8)
number_of_genres = len(dataset.genres_labels)

model = GenreClassifier(number_of_genres).to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initializing in a separate cell so we can easily add more epochs to the same run
epoch_number = 0
EPOCHS = 10

best_vloss = 1_000_000.0
best_model_state = None


for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)
    running_vloss = 0.0

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels, _ = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            voutputs = torch.sigmoid(voutputs)  # Convert logits to probabilities

            # Find the index of the max probability for each image
            predicted_indices = torch.argmax(voutputs, dim=1)
            predicted_labels = dataset.decode_indices_to_genres(predicted_indices)

            # You will need to implement the decode_indices_to_genres function
            # to convert indices to genre strings based on your dataset's encoding.

            original_labels = dataset.decode_label_to_string(vlabels)
            print(f"Original labels: {original_labels}")
            print(f"Predicted labels: {predicted_labels}")
            if predicted_labels == original_labels:
                # Print "yay" in green
                print("\033[92mYay\033[0m")
            else:
                print("\033[91mNope\033[0m")
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_model_state = model.state_dict().copy()

        model_path = "../../runs/genre_models/genre_folder.pt"
        torch.save(best_model_state, model_path)

    epoch_number += 1
