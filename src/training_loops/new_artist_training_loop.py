from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, Grayscale, RandomRotation, CenterCrop
from src.datasets.new_genre_dataset import *
from torch.utils.data import random_split

csv_file = 'path/to/your/csvfile.csv'
img_dir = 'path/to/your/imagedirectory/'
genre = 'Impressionism'
transform = Compose([
    Resize(size=(256, 256)),
    CenterCrop(size=(256, 256)),
    Grayscale(num_output_channels=1),
    ToTensor(),
    Normalize((0.5,), (0.5,)),  # Grayscale mean and std
])

# Create the full dataset
full_dataset = GenreDataset(csv_file=csv_file, img_dir=img_dir, genre=genre, transform=transform)

# Calculate the sizes for train and test datasets
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size

# Split the dataset into training and test datasets
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Example: Iterate over the train DataLoader
for images, labels in train_loader:
    # Process your images and labels for training
    pass

# Example: Iterate over the test DataLoader
for images, labels in test_loader:
    # Process your images and labels for testing
    pass
