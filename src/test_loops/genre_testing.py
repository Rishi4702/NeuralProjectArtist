import torch
import torchvision.transforms as transforms
from src.datasets.new_art_dataset import ArtDataset
from src.models.new_genre_classifier import *
from src.utils.dataloader import *
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomRotation, RandomHorizontalFlip, Grayscale
import torchvision.models as models
import torch.nn as nn
# Load the saved model
model_path = r"C:\Users\golur\PycharmProjects\NeuralProjectArtist\runs\genre_models\genre_model1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you're using a modified ResNet model, but replace with your actual model


train_transform = Compose([
    Resize(size=(256, 256)),
    CenterCrop(size=(256, 256)),
    Grayscale(num_output_channels=1),
    RandomRotation(degrees=(-10, 10)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.5,), (0.5,)),  # Grayscale mean and std
])
# Test dataset and loader
# Define your test_transform (similar to valid_transform)
test_transform = Compose([
    Resize(size=(256, 256)),
    CenterCrop(size=(256, 256)),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize((0.5,), (0.5,)),
])

test_dataset = ArtDataset(csv_file=r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\csv\artists.csv',
                     img_dir=r'C:\Users\golur\PycharmProjects\NeuralProjectArtist\dataset_files\resized', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=362)
train_dataset, valid_dataset = split_dataset(test_dataset, train_size=0.1, train_transform=train_transform, valid_transform=test_transform)
training_loader, validation_loader = get_data_loaders(train_dataset, valid_dataset, batch_size=62)
number_of_genres = test_dataset.num_gen()
model = modify_resnet_model(models.resnet18(pretrained=True), num_classes=number_of_genres)  # Update num_classes as needed
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Testing loop
total_predictions = 0
correct_predictions = 0
with torch.no_grad():
    for images, genres in tqdm(test_loader, desc="Testing"):
        images, genres = images.to(device), genres.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == genres).sum().item()
        total_predictions += genres.size(0)

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f"Test Accuracy: {accuracy:.4f}")
