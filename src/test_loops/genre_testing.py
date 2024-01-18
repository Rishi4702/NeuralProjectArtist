import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

from src.datasets.art_dataset import ArtDataset
from src.models.new_genre_classifier import *
from src.utils.dataloader import *

model_path = "../../runs/genre_models/genre_model1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Grayscale mean and std
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

test_dataset = ArtDataset(
    csv_file="../../dataset_files/csv/artists.csv",
    img_dir="../../dataset_files/resized",
    transform=test_transform,
    data_type="test",
)
test_loader = DataLoader(test_dataset, batch_size=62)
train_dataset, valid_dataset = split_dataset(
    test_dataset,
    train_size=0.3,
    train_transform=train_transform,
    valid_transform=test_transform,
)
training_loader, validation_loader = get_data_loaders(
    train_dataset, valid_dataset, batch_size=62
)
number_of_genres = test_dataset.num_gen()
model = modify_resnet_model(
    models.resnet18(pretrained=True), num_classes=number_of_genres
)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()


total_predictions = 0
correct_predictions = 0
with torch.no_grad():
    for images, genres in tqdm(test_loader, desc="Testing"):
        images, genres = images.to(device), genres.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == genres).sum().item()
        total_predictions += genres.size(0)

accuracy = correct_predictions / total_predictions
print(f"Test Accuracy: {accuracy:.4f}")
