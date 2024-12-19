import torch
from torch import nn

def encoder_block(in_channels, out_channels, kernel_size=3, padding=1):
    '''
    Создает блок энкодера: conv -> batchnorm -> relu -> conv -> batchnorm -> relu -> maxpool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2)
    )

def decoder_block(in_channels, out_channels, kernel_size=3, padding=1):
    '''
    Создает блок декодера: upsample -> conv -> batchnorm -> relu -> conv -> batchnorm -> relu
    '''
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = encoder_block(in_channels, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = decoder_block(1024 + 512, 512)
        self.decoder3 = decoder_block(512 + 256, 256)
        self.decoder2 = decoder_block(256 + 128, 128)
        self.decoder1 = decoder_block(128 + 64, 64)

        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(torch.cat([bottleneck, enc4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))

        return self.final_layer(dec1)

def create_model(in_channels, out_channels):
    '''
    Функция для создания модели UNet
    '''
    return UNet(in_channels, out_channels)
