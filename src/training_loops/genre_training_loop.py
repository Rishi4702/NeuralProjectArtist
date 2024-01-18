import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torchvision.transforms import (CenterCrop, Compose, Grayscale, Normalize,
                                    RandomHorizontalFlip, RandomRotation,
                                    Resize, ToTensor)
from tqdm import tqdm

from src.datasets.art_dataset import ArtDataset
from src.models.new_genre_classifier import *
from src.utils.dataloader import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../../runs/genre_models/genre_model1.pth"

GREEN = "\033[92m"
RED = "\033[91m"
ENDC = "\033[0m"
train_losses = []
valid_losses = []
accuracies = []

train_transform = Compose(
    [
        Resize(size=(256, 256)),
        CenterCrop(size=(256, 256)),
        Grayscale(num_output_channels=1),
        RandomRotation(degrees=(-10, 10)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ]
)

valid_transform = Compose(
    [
        Resize(size=(256, 256)),
        CenterCrop(size=(256, 256)),
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ]
)

# Dataset and Data Loaders
dataset = ArtDataset(
    csv_file="../../dataset_files/csv/artists.csv",
    img_dir="../../dataset_files/resized",
)
train_dataset, valid_dataset = split_dataset(
    dataset,
    train_size=0.8,
    train_transform=train_transform,
    valid_transform=valid_transform,
)
training_loader, validation_loader = get_data_loaders(
    train_dataset, valid_dataset, batch_size=64
)
number_of_genres = dataset.num_gen()

model = models.resnet18(pretrained=True)
model = modify_resnet_model(model, number_of_genres)

# Loss function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = model.to(device)
scaler = GradScaler()

EPOCHS = 7
epoch_number = 0
best_vloss = float("inf")
best_model_state = None

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch + 1))

    model.train(True)
    total_loss = 0.0
    for i, data in enumerate(tqdm(training_loader, desc=f"Epoch {epoch + 1}/Training")):
        images, genres = data
        images = images.to(device)
        genres = genres.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, genres)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(training_loader)
    print("Average training loss: {:.4f}".format(avg_loss))

    model.eval()
    total_vloss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for vdata in tqdm(validation_loader, desc=f"Epoch {epoch + 1}/Validation"):
            vimages, vgenres = vdata
            vimages = vimages.to(device)
            vgenres = vgenres.to(device).long()

            voutputs = model(vimages)
            vloss = loss_fn(voutputs, vgenres)
            total_vloss += vloss.item()

            _, predicted_indices = torch.max(voutputs, 1)
            correct_predictions += (predicted_indices == vgenres).sum().item()
            total_predictions += vgenres.size(0)

    avg_vloss = total_vloss / len(validation_loader)
    accuracy = correct_predictions / total_predictions
    print("Validation loss: {:.4f}, Accuracy: {:.4f}".format(avg_vloss, accuracy))
    train_losses.append(avg_loss)
    valid_losses.append(avg_vloss)
    accuracies.append(accuracy)
    # Save best model
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, model_path)

    epoch_number += 1

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Training Loss")
plt.plot(epochs_range, valid_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, accuracies, label="Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
