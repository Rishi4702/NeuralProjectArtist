import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ColorJitter
import torch
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torchvision.transforms as transforms
from src.datasets.new_art_dataset import ArtDataset
from src.utils.dataloader import *
from src.models.new_genre_classifier import GenreClassifier
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomRotation, RandomHorizontalFlip, ColorJitter, Grayscale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:\Users\golur\PycharmProjects\NeuralProjectArtist\runs\genre_models\genre_model.pth"
# ANSI escape codes for colors
GREEN = "\033[92m"
RED = "\033[91m"
ENDC = "\033[0m"  # Reset to default color
train_transform = Compose([
    Resize(size=(256, 256)),
    CenterCrop(size=(256, 256)),
    Grayscale(num_output_channels=1),
    RandomRotation(degrees=(-10, 10)),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ToTensor(),
    Normalize((0.5,), (0.5,)),  # Grayscale mean and std
])

valid_transform = Compose([
    Resize(size=(256, 256)),
    CenterCrop(size=(256, 256)),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize((0.5,), (0.5,)),  # Grayscale mean and std
])


dataset = ArtDataset(csv_file=r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\csv\artist.csv',
                     img_dir=r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\resized')
# Assuming other necessary imports are already done
train_dataset, valid_dataset = split_dataset(dataset, train_size=0.8, train_transform=train_transform, valid_transform=valid_transform)

# Create data loaders
training_loader, validation_loader = get_data_loaders(train_dataset, valid_dataset, batch_size=62)
number_of_genres = dataset.num_gen()
# model = models.resnet18(pretrained=True).to(device)
model = GenreClassifier(number_of_genres).to(device)
# Update the loss function
loss_fn = nn.CrossEntropyLoss()
# Update the optimizer if needed, e.g., using Adam instead of SGD
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
# scheduler = StepLR(optimizer, step_size=5, gamma=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

EPOCHS = 25
epoch_number = 0
best_vloss = float('inf')
best_model_state = None
for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch + 1))

    model.train(True)
    total_loss = 0.0

    for i, data in enumerate(tqdm(training_loader, desc=f"Epoch {epoch + 1}/Training")):
        images, genres = data
        images, genres = data
        images = images.to(device)
        genres = genres.to(device).long()  # Ensure genres are long type for CrossEntropyLoss

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, genres)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(training_loader)
    print("Average training loss: {:.4f}".format(avg_loss))
    scheduler.step(avg_loss)
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

            for i in range(len(predicted_indices)):
                predicted_label = predicted_indices[i].item()
                true_label = vgenres[i].item()
                predicted_label_decoded = dataset.decode_label_to_string(predicted_label)
                true_label_decoded = dataset.decode_label_to_string(true_label)

                if predicted_label == true_label:
                    colored_text = GREEN + "Yay" + ENDC
                else:
                    colored_text = RED + "Nope" + ENDC

                print(f"Predicted Label: {predicted_label_decoded}, True Label: {true_label_decoded}, Result: {colored_text}")

    avg_vloss = total_vloss / len(validation_loader)
    accuracy = correct_predictions / total_predictions
    print("Validation loss: {:.4f}, Accuracy: {:.4f}".format(avg_vloss, accuracy))

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_model_state = model.state_dict().copy()
        torch.save(best_model_state, model_path)

    epoch_number += 1
