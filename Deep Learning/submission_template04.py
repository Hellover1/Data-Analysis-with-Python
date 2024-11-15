import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Определяем слои сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)  # 3 фильтра размера (5, 5)

ConvNet(
  (conv1): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1))
  (pool2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=180, out_features=100, bias=True)
  (fc2): Linear(in_features=100, out_features=10, bias=True)
)    
    def forward(self, x):
        ...

def create_model():
    return ConvNet()
