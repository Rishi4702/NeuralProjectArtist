import warnings

from torch.utils.data import DataLoader, random_split

warnings.warn("I am UserWarning", UserWarning)
warnings.warn("I am FutureWarning", FutureWarning)


def get_data_loaders(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_dataloader, test_dataloader
