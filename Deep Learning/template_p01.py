import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ef encoder_block(in_channels, out_channels, kernel_size, padding):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    return block

def decoder_block(in_channels, out_channels, kernel_size, padding):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

         self.encoder = nn.Sequential(
            encoder_block(3, 64, 3, 1),
            encoder_block(64, 128, 3, 1),
            encoder_block(128, 256, 3, 1)
        )

        self.decoder = nn.Sequential(
            decoder_block(256, 128, 3, 1),
            decoder_block(128, 64, 3, 1),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        # downsampling 
        latent = self.encoder(x)

        # upsampling
        reconstruction = self.decoder(latent)

        return reconstruction



def create_model():
    return Autoencoder()
