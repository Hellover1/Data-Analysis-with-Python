import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

    return block

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Добавляем слои энкодера
        self.encoder = nn.Sequential(
            encoder_block(3, 64, kernel_size=3, padding=1),
            encoder_block(64, 128, kernel_size=3, padding=1),
            encoder_block(128, 256, kernel_size=3, padding=1)
        )

        # Добавляем слои декодера
        self.decoder = nn.Sequential(
            decoder_block(256, 128, kernel_size=3, padding=1),
            decoder_block(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Для восстановления значений в диапазоне [-1, 1]
        )

    def forward(self, x):
        # Downsampling
        latent = self.encoder(x)

        # Upsampling
        reconstruction = self.decoder(latent)

        return reconstruction

def create_model():
    return Autoencoder()
