import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        """
        the first conv layer:
        in_channels:1
        out_channels:6
        kerner_size:5*5
        stride:1
        padding:2
        """
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        """
        the second conv layer:
        in_channels:6
        out_channels:16
        kerner_size:5*5
        stride:1
        padding:0
        """
        self.conv2 = nn.Conv2d(6, 16 , 5, stride=1, padding=2)
        """
        the first fully connected layer:
        in_features:16*5*5
        out_features:120
        """
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        """
        the first fully connected layer:
        in_features:120
        out_features:84
        """
        self.fc2 = nn.Linear(120, 84)
        """
        the first fully connected layer:
        in_features:84
        out_features:10
        """
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # the first pooling layer
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride = 2)
        # the second pooling layer
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride = 2)

        # Convert multidimensional data to 2D data
        x = x.view(-1, self.num_flat_features(x))
        # print("x "+str(x.shape))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    # Convert multidimensional data to 2D data
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
