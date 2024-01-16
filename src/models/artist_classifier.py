import torch
import torch.nn as nn
import torch.nn.functional as F

class ArtistClassifier(nn.Module):
    def __init__(self, number_of_artists):
        super(ArtistClassifier, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 62 * 62, 120)  # Adjust based on your input image dimensions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, number_of_artists)

    def forward(self, x):
        x = x.float()
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

