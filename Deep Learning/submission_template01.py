import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Устройство: CPU или GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Определение UNet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.encoder_block(in_channels, 64)
        self.encoder2 = self.encoder_block(64, 128)
        self.encoder3 = self.encoder_block(128, 256)
        self.encoder4 = self.encoder_block(256, 512)
        
        self.decoder4 = self.decoder_block(512, 256)
        self.decoder3 = self.decoder_block(256 + 256, 128)
        self.decoder2 = self.decoder_block(128 + 128, 64)
        self.decoder1 = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.decoder4(enc4)
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))
        return dec1

# Подготовка данных
train_data = torch.rand(50, 3, 128, 128)  # Пример входных данных (3 канала)
train_targets = torch.randint(0, 5, (50, 128, 128))  # Пример сегментационных масок (5 классов)

val_data = torch.rand(10, 3, 128, 128)
val_targets = torch.randint(0, 5, (10, 128, 128))

train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=8, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data, val_targets), batch_size=8)

# Параметры модели
model = UNet(in_channels=3, out_channels=5).to(device)
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Функция обучения
def train(model, opt, loss_fn, epochs, train_loader, val_loader):
    for epoch in range(epochs):
        print(f'* Epoch {epoch + 1}/{epochs}')
        
        # Training
        model.train()
        avg_train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            opt.zero_grad()
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch.long())
            loss.backward()
            opt.step()
            avg_train_loss += loss.item() / len(train_loader)
        print(f'avg train loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        avg_val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                Y_pred = model(X_batch)
                loss = loss_fn(Y_pred, Y_batch.long())
                avg_val_loss += loss.item() / len(val_loader)
        print(f'avg val loss: {avg_val_loss:.4f}')

# Запуск обучения
train(model, opt, loss, 10, train_loader, val_loader)

