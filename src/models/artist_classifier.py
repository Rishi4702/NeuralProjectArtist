import torch
import torch.nn as nn
import torch.nn.functional as F


class ArtistClassifier(nn.Module):
    def __init__(self, number_of_artists):
        super(ArtistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(
            16 * 62 * 62, 120
        )
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, number_of_artists)

    def forward(self, x):
        x = x.float()
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
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

def modify_resnet_model(model, num_classes):
    first_conv_layer = model.conv1
    model.conv1 = nn.Conv2d(
        1,
        first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=False,
    )

    # Modify the fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
