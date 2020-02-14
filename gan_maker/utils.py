"""
@author: picost

Module containing utility function used to create CNNs with batch norm.

"""
import torch.nn as nn

def conv_BN(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
            batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.

    Args:
    -----

        in_channels (int): number of channels in conv layer inputs
        out_channels (int): number of channels in conv layer outputs
        kernel_size (int/tuple, optional): size of convlution kernel to be used
        stride (int, optional): convolution stride
        padding (int, optional): convolution padding
        batch_norm (bool, optional): If True (default) a batchnormalization layer
            is added following the convolution layer.

    Returns
    -------

        nn.Module instance representing a convolutional layer, eventually composed
            with a batch normalisation layer

    Note:
    -----

        If batch normalization is used, then no bias is used in the convolutional layer
            (redundant/adevrsarial with the shift parameter in the batch normalization)
        Default convolutional kernel, stride and padding values are chosen in order to
            downscale a picture with factor 2 if original size is even.

    """

    if batch_norm:
        layers = []
        conv_layer = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        layers.append(conv_layer)
        layers.append(nn.BatchNorm2d(out_channels))
        new_mod = nn.Sequential(*layers)
    else:
        new_mod = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=True)
    return new_mod


def conv_transpose_BN(in_channels, out_channels, kernel_size=4, stride=2,
                      padding=1, batch_norm=True):
    """Creates a transposed convolutional layer, with optional batch normalization.

    Args:
    -----

        in_channels (int): number of channels in conv layer inputs
        out_channels (int): number of channels in conv layer outputs
        kernel_size (int/tuple, optional): size of convlution kernel to be used
        stride (int, optional): convolution stride
        padding (int, optional): convolution padding
        batch_norm (bool, optional): If True (default) a batchnormalization layer
            is added following the convolution layer.

    Returns
    -------

        nn.Module instance representing a transposed convolutional layer,
        eventually composed with a batch normalisation layer

    Note:
    -----

        If batch normalization is used, then no bias is used in the convolutional
            layer (redundant/adevrsarial with the shift parameter in the batch
            normalization)
        Default convolutional kernel, stride and padding values are chosen in
            order to downscale a picture with factor 2 if original size is even.

    """

    if batch_norm:
        layers = []
        conv_layer = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False)
        layers.append(conv_layer)
        layers.append(nn.BatchNorm2d(out_channels))
        new_mod = nn.Sequential(*layers)
    else:
        new_mod = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=True)
    return new_mod