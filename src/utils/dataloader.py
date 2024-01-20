from torch.utils.data import DataLoader, random_split


def get_data_loaders(train_dataset, valid_dataset, batch_size=62):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


def split_dataset(dataset, train_size, train_transform, valid_transform):
    train_len = int(train_size * len(dataset))
    valid_len = len(dataset) - train_len
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = valid_transform

    return train_dataset, valid_dataset
