import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Encoder block function
def encoder_block(in_channels, out_channels, kernel_size, padding):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    return block

# Decoder block function
def decoder_block(in_channels, out_channels, kernel_size, padding):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            encoder_block(3, 64, 3, 1),
            encoder_block(64, 128, 3, 1),
            encoder_block(128, 256, 3, 1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            decoder_block(256, 128, 3, 1),
            decoder_block(128, 64, 3, 1),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Downsampling (encoder)
        latent = self.encoder(x)

        # Upsampling (decoder)
        reconstruction = self.decoder(latent)

        return reconstruction

# Function to create the model
def create_model():
    return Autoencoder()
