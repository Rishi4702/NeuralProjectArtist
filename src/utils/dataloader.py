from torch.utils.data import DataLoader, random_split

def get_data_loaders(train_dataset, valid_dataset, batch_size=62):
    """
    Creates data loaders for the training and validation datasets.

    Args:
    - train_dataset: Dataset object for training.
    - valid_dataset: Dataset object for validation.
    - batch_size (int, optional): Batch size for the data loaders. Defaults to 500.

    Returns:
    - Tuple of (train_dataloader, valid_dataloader).
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader

def split_dataset(dataset, train_size, train_transform, valid_transform):
    """
    Splits the dataset into training and validation datasets and applies different transforms.

    Args:
    - dataset: The original dataset to split.
    - train_size: Fraction of the dataset to use for training (0 to 1).
    - train_transform: Transform to apply to the training dataset.
    - valid_transform: Transform to apply to the validation dataset.

    Returns:
    - Tuple of (train_dataset, valid_dataset).
    """
    train_len = int(train_size * len(dataset))
    valid_len = len(dataset) - train_len
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

    # Apply the respective transforms
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = valid_transform

    return train_dataset, valid_dataset
