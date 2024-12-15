import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def encoder_block(in_channels, out_channels, kernel_size=3, padding=1):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)  # Уменьшаем размер карты активации в 2 раза
    )
    return block

def decoder_block(in_channels, out_channels, kernel_size=3, padding=1):
    '''
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # Увеличиваем размер карты активации в 2 раза
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            encoder_block(3, 64),
            encoder_block(64, 128),
            encoder_block(128, 256),
        )

        # Decoder
        self.decoder = nn.Sequential(
            decoder_block(256, 128),
            decoder_block(128, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),  # Последний слой увеличивает размер карты
            nn.Tanh()  # Для нормализации значений пикселей
        )

    def forward(self, x):
        # Downsampling 
        latent = self.encoder(x)

        # Upsampling
        reconstruction = self.decoder(latent)

        return reconstruction


def create_model():
    return Autoencoder()
