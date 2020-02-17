"""
Created on Fri Feb  7 14:49:43 2020

This module implements a ResidualBlock module, as from presented in the paper
from He & al. (arXiv:1512.03385v1).

This is mostly taken from the module implemented in the cycle-gan class.

@author: picost
"""
import torch.nn as nn

from.utils import conv

class ResidualBlock(nn.Module):
    """Defines a residual block with two layers.

       This residual block is a successsion of two convolutional layers with
       same input/output shapes, combined with a direct additive bypass from
       input to output.

       This kind of structure has proven to lead to more easily optimized
       networks in some situations (see origina paper).

    """

    def __init__(self, n_channels, kernel_size=3):
        """Initialize the block according the given number of channels and
        kernel size.

        All convolutional layers use batch-normalization

        Args :
        ------

            n_channels (int): number of channels in the convolutiona layers
            kernel_size (int, optional): Size of kernel in the convolutional
                layers. Other parameters are deduced so that the input and
                output shape remain the same.

        Returns :
        ---------

            initialized self

        """
        padding = int(kernel_size / 2)
        super(ResidualBlock, self).__init__()
        self.conv_layer1 = conv(in_channels=n_channels, out_channels=n_channels,
                                kernel_size=kernel_size, stride=1, padding=padding,
                                batch_norm=True)
        self.conv_layer2 = conv(in_channels=n_channels, out_channels=n_channels,
                               kernel_size=kernel_size, stride=1, padding=padding,
                               batch_norm=True)

    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = nn.functional.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2