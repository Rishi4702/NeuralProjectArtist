import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelPaintingCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelPaintingCNN, self).__init__()
        # Grayscale images have 1 input channel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)  # Adjust based on your image size
        self.fc2 = nn.Linear(512, num_classes)  # One output per class
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Use sigmoid for multi-label classification
        return x
