#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:43:36 2020

@author: picost
"""
import torch.nn as nn

from ..residuals import ResidualBlock

class Generator(nn.Module):

    def __init__(self, z_size, conv_dim, n_res=3):
        """
        Initialize the Generator Module

        Args:
        -----

            z_size (int): The length of the input latent vector, z
            conv_dim (int): The depth of the inputs to the *last* transpose
                convolutional layer

        Returns:
        --------

            A generator to be trained

        """
        super(Generator, self).__init__()

        res_layers = []
        for k_res in range(0, n_res):
            res_layers.append(ResidualBlock(n_res_chan))

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior

        return x