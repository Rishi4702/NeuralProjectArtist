from torch.utils.data import DataLoader
from src.Dataloader.ArtDataset import ArtDataset
from torchvision import transforms
from torch.utils.data import random_split


def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[255, 255]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ArtDataset(csv_file='../Dataset/artists.csv', img_dir='../Dataset/resized', transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_dataloader, test_dataloader