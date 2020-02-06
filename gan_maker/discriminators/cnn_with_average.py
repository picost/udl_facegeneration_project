#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:57:59 2020

@author: picost
"""
import torch.nn as nn

from ..utils import conv_BN as conv

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()
        # --- Convolutional layer (in calling order ) ---
        # 32 * 32 * 3
        self.conv1 = conv(3, conv_dim, kernel_size=4, stride=2, padding=1, 
                          batch_norm=False)
        # 16 * 16 * conv_dim
        self.conv2 = conv(conv_dim, conv_dim * 2, kernel_size=4, stride=2, 
                          padding=1, batch_norm=True)
        # 8 * 8 * conv_dim*2
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, 
                          padding=1, batch_norm=True)
        # 4 * 4 * conv_dim*4
        self.conv4 = conv(conv_dim * 4 , conv_dim * 8,  kernel_size=2, stride=1, 
                          padding=1, batch_norm=False)
        # 
        # --- Other layers
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        
        return x