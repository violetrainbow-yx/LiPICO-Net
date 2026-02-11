import torch
import torch.nn as nn
from torch.nn import functional as F

class LiPICO_Net(nn.Module):
    def __init__(self, measurement_rate):
        super(LiPICO_Net, self).__init__()
        self.fc1 = nn.Linear(int(measurement_rate*1024), 1024)
        self.conv1 = nn.Conv2d(1, 16, 4, 1)
        self.conv2 = nn.Conv2d(16, 16, 1, 1)
        self.conv3 = nn.Conv2d(16, 16, 4, 1)
        self.pad = nn.ZeroPad2d((1, 2, 1, 2))
        self.intermediate_output = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 32, 32)
        x = x.unsqueeze(1)
        self.intermediate_output = x
        x = self.pad(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pad(x)
        x = self.conv3(x)
        x = x.sum(dim=1, keepdim=True)
        x = x + self.intermediate_output
        return x

