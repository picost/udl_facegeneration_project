#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:43:36 2020

This module implements a particular generator for the face-generation case.

@author: picost
"""
import torch.nn as nn

from ..utils import conv_transpose_BN

class Generator(nn.Module):
    """This class implements a generator class which generates square images
    with 3 channels and pixel values ranging in [-1,1].

    The Generator is made of:

        - A Fully connected layer transforming an input vector to picture features
        - Two transposed convolutional layers upsacaling the feature with kernels
          of size 4 followed by batch normalization then leaky ReLU.
        - A transposed convolutional layer upscaling the features with kernel of
          size 4, 3 output channels, no batch normalization, followed by a Tanh
          activation function.


    """

    def __init__(self, z_size, conv_dim, img_size=32):
        """Initialize the Generator Module

        Args:
        -----

            z_size (int):
                The length of the input latent vector, z
            conv_dim (int):
                The depth of the inputs to the *last* transpose
                convolutional layer.
            img_size (int, optional):
                Size of x and y dimensions of the produced output image. This
                number must be a divisible by 8 (3 steps of upsacaling with
                factor 2)

        Returns:
        --------

            nn.Module
                A generator to be trained

        """
        super(Generator, self).__init__()
        # (img_size / 8)^2 * conv_dim * 4
        self._conv_dim = conv_dim
        self.im_size = img_size
        self._feat_size = img_size // 8
        self._in_conv_dim = conv_dim * 4
        in_conv_size = self._feat_size * self._feat_size * self._in_conv_dim
        self.fc_in = nn.Linear(z_size, in_conv_size)
        self.dconv1 = conv_transpose_BN(
                conv_dim * 4, conv_dim *2, kernel_size=4, stride=2, padding=1,
                batch_norm=True)
        self.dconv2 = conv_transpose_BN(
                conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1,
                batch_norm=True)
        self.dconv2 = conv_transpose_BN(
                conv_dim, 3, kernel_size=4, stride=2, padding=1,
                batch_norm=False)
        self.leak = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Return network output for the input batch

        Forward propagation of the neural network

        Args:
        -----

            x (Tensor):
                The input to the neural network with shape (batch_size,
                n_channels, img_size, img_size)

        Returns:
        --------

            Tensor :
                A 32x32x3 Tensor image as output if img_size is 32

        """
        feat = self.fc_in(x)
        feat = feat.view(x.shape[0], self._in_conv_dim, self._feat_size,
                         self._feat_size)
        feat = self.leak(self.dconv1(feat))
        feat = self.leak(self.dconv2(feat))
        feat = self.tanh(self.dconv3(feat))
        return feat