import torch
import torch.nn as nn
import torch.nn.functional as F


class GenreClassifier(nn.Module):
    def __init__(self, n_classes):
        super(GenreClassifier, self).__init__()
        # Convolutional layers for grayscale images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # Flatten the output for the dense layers
        x = x.view(-1, 64 * 16 * 16)
        # Dense layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
