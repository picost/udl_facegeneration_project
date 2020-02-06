#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:57:59 2020

@author: picost
"""
import torch.nn as nn

from ..utils import conv_BN as conv

class Discriminator(nn.Module):
    """A discriminator class to be applied on 2D images
    
    The discriminator structure is as follows:
        
        - The first layer is convolutional starting from 3 channels to the
          number provided at init. No batch normalization is used.
        - The two following layers increase the depths by a factor 2 while
          decreasing the other dimensions by the same factor.
        - ReLU is applied after each of these layers.
        - The last layer is used for classification. A kernel with size 3 is
          used with stride and padding of 1 to let the "2D size" unchanged while
          decreasing the number of channels to 1. No batch-norm.
        - The values in this last feature array are averages to provide the
          output classification value. This is inspired from a previous experiment
          in the cycle-gan exercise.
    
    """

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        
        Args:
        -----
        
            conv_dim: The depth of the first convolutional layer. The depth
                is then increased by a factor two at each layer.
                
                
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
        self.conv_classify = conv(conv_dim * 4 , 1,  kernel_size=3, stride=1, 
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
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv_classify(x)
        x = x.view(x.shape[0], -1)
        out = x.mean(dim=1, keepdim=True)
        return out