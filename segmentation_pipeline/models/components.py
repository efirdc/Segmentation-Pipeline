from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


def prod(sequence):
    out = 1
    for elem in sequence:
        out *= elem
    return out


class Block3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            conv_class=nn.Conv3d,
            conv_params=None,
            normalization_class=nn.BatchNorm3d,
            normalization_params=None,
            activation_class=nn.ReLU,
            activation_params=None,
            residual=False,
            residual_params=None,
            dropout_p=0.0,
            num_convs=2,
    ):
        super().__init__()

        if conv_params is None:
            conv_params = {'bias': False, 'kernel_size': 3, 'padding': 1}
        if normalization_params is None:
            normalization_params = {}
        if activation_params is None:
            activation_params = {'inplace': True}
        if residual_params is None:
            residual_params = {'bias': True, 'kernel_size': 3, 'padding': 1}

        self.residual = residual
        if self.residual:
            self.res_conv = conv_class(in_channels, out_channels, **residual_params)

        parts = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            parts.append((f'conv{i}', conv_class(in_ch, out_channels, **conv_params)))
            if normalization_class is not None:
                parts.append((f'norm{i}', normalization_class(out_channels, **normalization_params)))
            if activation_class is not None:
                parts.append((f'activation{i}', activation_class(**activation_params)))
        self.layers = nn.Sequential(OrderedDict(parts))

        self.dropout = None
        if dropout_p != 0.0:
            self.dropout = nn.Dropout3d(p=dropout_p)

    def forward(self, x):
        x_in = x

        x = self.layers(x)

        if self.residual:
            x = self.res_conv(x_in) + x

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class WSConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.kwargs = kwargs

    def forward(self, x):
        weight = self.weight
        weight = weight - weight.mean(dim=(1, 2, 3, 4), keepdim=True)
        weight = weight / (weight.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-5)

        x = F.conv3d(x, weight, **self.kwargs)

        return x


class BlurConv3d(nn.Conv3d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_standardization=False,
            **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.weight_standardization = weight_standardization

        kernel = torch.ones(out_channels, 1, 2, 2, 2)
        kernel = kernel / 8
        self.register_buffer('kernel', kernel)

        # Volume is shrinking by stride^3
        self.kernel = self.kernel / prod(self.stride)
        self.kwargs = kwargs

    def forward(self, x):
        weight = self.weight

        if self.weight_standardization:
            weight = weight - weight.mean(dim=(1, 2, 3, 4), keepdim=True)
            weight = weight / (weight.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-5)

        weight = F.conv3d(weight, self.kernel, padding=1, groups=self.in_channels)
        x = F.conv3d(x, weight, **self.kwargs)

        return x


class BlurConvTranspose3d(nn.ConvTranspose3d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_standardization=False,
            **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.weight_standardization = weight_standardization

        kernel = torch.ones(out_channels, 1, 2, 2, 2)
        kernel = kernel / torch.sum(kernel)
        self.register_buffer('kernel', kernel)

        # Volume is growing by stride^3
        self.kernel = self.kernel * prod(self.stride)
        self.kwargs = kwargs

    def forward(self, x, output_size=None):
        weight = self.weight

        if self.weight_standardization:
            weight = weight - weight.mean(dim=(1, 2, 3, 4), keepdim=True)
            weight = weight / (weight.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-5)

        weight = F.conv3d(weight, self.kernel, padding=1, groups=self.in_channels)
        x = F.conv_transpose3d(x, weight, **self.kwargs)

        return x