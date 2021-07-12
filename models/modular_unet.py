from typing import Union, Sequence, Dict, Optional

import torch
from torch import nn

from utils import is_sequence
from .components import Block3d
from .model_utils import filter_kwargs


class ModularUNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            filters: Union[int, Sequence[int]],
            depth: int,
            block_class: nn.Module = Block3d,
            block_params: Optional[Dict] = None,
            upsample_class: nn.Module = nn.Upsample,
            upsample_params: Optional[Dict] = None,
            downsample_class: nn.Module = nn.AvgPool3d,
            downsample_params: Optional[Dict] = None,
            out_conv_class: nn.Module = nn.Conv3d,
            out_conv_params: Optional[Dict] = None,
            hypothesis_class: nn.Module = nn.Softmax,
            hypothesis_params: Optional[Dict] = None,
    ):
        super().__init__()

        if isinstance(filters, int):
            filters = [filters] * depth
        elif is_sequence(filters) and len(filters) != depth:
            raise ValueError(f"Sequence of filters {filters} does not match depth {depth}")

        if block_params is None:
            block_params = {}
        if upsample_params is None:
            upsample_params = {'scale_factor': 2, 'mode': 'trilinear', 'align_corners': True}
        if downsample_params is None:
            downsample_params = {'kernel_size': 2, 'stride': 2, 'count_include_pad': False}
        if out_conv_params is None:
            out_conv_params = {'in_channels': filters[0], 'out_channels': out_channels, 'kernel_size': 3,
                               'padding': 1}
        if hypothesis_params is None:
            hypothesis_params = {"dim": 1}

        self.depth = depth

        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(block_class(in_channels, filters[0], **block_params))
        for i in range(1, depth):
            down_block = block_class(filters[i - 1], filters[i], **block_params)
            self.down_blocks.append(down_block)

        self.downsampling = nn.ModuleList()
        for i in range(depth - 1):
            downsample_params.update(filter_kwargs(
                downsample_class,
                in_channels=filters[i],
                out_channels=filters[i],
                channels=filters[i]
            ))
            downsample = downsample_class(**downsample_params)
            self.downsampling.append(downsample)

        self.up_blocks = nn.ModuleList()
        for i in range(depth - 1):
            up_block = block_class(filters[i] + filters[i + 1], filters[i], **block_params)
            self.up_blocks.append(up_block)

        self.upsampling = nn.ModuleList()
        for i in range(1, depth):
            upsample_params.update(filter_kwargs(
                upsample_class,
                in_channels=filters[i],
                out_channels=filters[i],
                channels=filters[i]
            ))
            upsample = upsample_class(**upsample_params)
            self.upsampling.append(upsample)

        self.out_conv = out_conv_class(**out_conv_params)
        self.hypothesis = hypothesis_class(**hypothesis_params)

    def forward(self, x):

        down_block_output = []
        for i in range(self.depth):
            x = self.down_blocks[i](x)
            if i != (self.depth - 1):
                down_block_output.append(x)
                x = self.downsampling[i](x)

        up_parts = list(zip(self.upsampling, self.up_blocks, down_block_output))
        for upsample, up_block, x_skip in reversed(up_parts):
            x = upsample(x)
            x = up_block(torch.cat([x, x_skip], dim=1))

        x = self.out_conv(x)
        x = self.hypothesis(x)

        return x
