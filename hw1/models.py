import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x
