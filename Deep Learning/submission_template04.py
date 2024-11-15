import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
    super(ConvNet, self).__init__()

        # Определяем слои сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)  # 3 фильтра размера (5, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # maxpool1 с ядром размера 2
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)  # 5 фильтров размера (3, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # maxpool2 с ядром размера 2

        self.flatten = nn.Flatten()  # Для преобразования тензора в вектор

        self.fc1 = nn.Linear(in_features=5 * 6 * 6, out_features=100)  # 100 нейронов на выходе
        self.fc2 = nn.Linear(in_features=100, out_features=10)  # 10 нейронов на выходе

    def forward(self, x):
        # Применяем свертки и активации
        x = self.conv1(x)
        x = F.relu(x)  # Применяем ReLU
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)  # Применяем ReLU
        x = self.pool2(x)

        x = self.flatten(x)  # Преобразуем в вектор
        x = self.fc1(x)
        x = F.relu(x)  # Применяем ReLU к полносвязному слою
        x = self.fc2(x)

def create_model():
    return ConvNet()
