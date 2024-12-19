import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def softmax(vector):
def encoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Применяем формулу Multiplicative Attention
    softmax_vector = softmax(decoder_hidden_state.T @ W_mult @ encoder_hidden_states)
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T
    return attention_vector
    # Реализуйте блок вида conv -> relu -> max_pooling. 
    # Параметры слоя conv заданы параметрами функции encoder_block. 
    # MaxPooling должен быть с ядром размера 2.
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    return block
def decoder_block(in_channels, out_channels, kernel_size, padding):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    блок, который принимает на вход карты активации с количеством каналов in_channels, 
    и выдает на выход карты активации с количеством каналов out_channels
    kernel_size, padding — параметры conv слоев внутри блока
    '''
    # Применяем формулу Additive Attention
    softmax_vector = softmax(v_add.T @ np.tanh(W_add_enc @ encoder_hidden_states + W_add_dec @ decoder_hidden_state))
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T
    return attention_vector
    # Реализуйте блок вида conv -> relu -> upsample. 
    # Параметры слоя conv заданы параметрами функции encoder_block. 
    # Upsample должен быть со scale_factor=2. Тип upsampling (mode) можно выбрать любым.
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
    )
    return block
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        параметры: 
            - in_channels: количество каналов входного изображения
            - out_channels: количество каналов выхода нейросети
        '''
        super().__init__()
        self.enc1_block = encoder_block(in_channels, 32, 7, 3)
        self.enc2_block = encoder_block(32, 64, 3, 1)
        self.enc3_block = encoder_block(64, 128, 3, 1)
        self.dec1_block = decoder_block(128, 64, 3, 1)
        self.dec2_block = decoder_block(64 + 64, 32, 3, 1)
        self.dec3_block = decoder_block(32 + 32, out_channels, 3, 1)
    def forward(self, x):
        # downsampling part
        enc1 = self.enc1_block(x)
        enc2 = self.enc2_block(enc1)
        enc3 = self.enc3_block(enc2)
        dec1 = self.dec1_block(enc3)
        # из-за skip connection dec2 должен принимать на вход сконкатенированные карты активации
        # из блока dec1 и из блока enc2. 
        # конкатенация делается с помощью torch.cat
        dec2 = self.dec2_block(torch.cat([dec1, enc2], 1))
        # из-за skip connection dec3 должен принимать на вход сконкатенированные карты активации
        # из блока dec2 и из блока enc1. 
        # конкатенация делается с помощью torch.cat
        dec3 = self.dec3_block(torch.cat([dec2, enc1], 1))
        return dec3
def create_model(in_channels, out_channels):
    # your code here
    # return model instance (None is just a placeholder)
    return UNet(in_channels, out_channels)
