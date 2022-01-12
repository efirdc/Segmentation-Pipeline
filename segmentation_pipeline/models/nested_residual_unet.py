import torch
from torch import nn
from typing import Optional, Dict


class NestedResUNet(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_ch, out_ch, residual=False, dropout_p=0.0):
            super().__init__()
            self.residual = residual
            self.out_ch = out_ch

            conv_class = nn.Conv3d
            conv_params = dict(kernel_size=3, padding=1)

            if self.residual:
                self.res_conv = conv_class(in_ch, out_ch, **conv_params)

            self.conv1 = conv_class(in_ch, out_ch, bias=False, **conv_params)
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.activation1 = nn.ReLU(inplace=True)
            self.conv2 = conv_class(out_ch, out_ch, bias=False, **conv_params)
            self.bn2 = nn.BatchNorm3d(out_ch)
            self.activation2 = nn.ReLU(inplace=True)

            self.dropout = None
            if dropout_p != 0.0:
                self.dropout = nn.Dropout3d(p=dropout_p)

        def forward(self, x):
            x_in = x

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.activation2(x)

            if self.residual:
                x = self.res_conv(x_in) + x

            if self.dropout is not None:
                x = self.dropout(x)

            return x

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            filters: int,
            dropout_p: float = 0.0,
            hypothesis_class: nn.Module = nn.Softmax,
            hypothesis_params: Optional[Dict] = None,
    ):
        super().__init__()

        if hypothesis_params is None:
            hypothesis_params = {"dim": 1}

        self.dropout = None
        if dropout_p != 0.0:
            self.dropout = nn.Dropout3d(p=dropout_p)

        self.down = nn.AvgPool3d(kernel_size=2, stride=2, count_include_pad=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        block_params = dict(dropout_p=dropout_p)

        self.conv0_0 = self.Block(input_channels, filters, **block_params, residual=True)
        self.conv1_0 = self.Block(filters, filters, **block_params)
        self.conv0_1 = self.Block(filters*2, filters, **block_params, residual=True)

        self.conv2_0 = self.Block(filters, filters, **block_params)
        self.conv1_1 = self.Block(filters*3, filters, **block_params)
        self.conv0_2 = self.Block(filters*2, filters, **block_params, residual=True)

        self.conv3_0 = self.Block(filters, filters, **block_params)
        self.conv2_1 = self.Block(filters*3, filters, **block_params)
        self.conv1_2 = self.Block(filters*3, filters, **block_params)
        self.conv0_3 = self.Block(filters*2, filters, **block_params, residual=True)

        self.out_conv = nn.Conv3d(filters, output_channels, kernel_size=3, padding=1)
        self.hypothesis = hypothesis_class(**hypothesis_params)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.down(x0_0))
        x0_1 = self.conv0_1(torch.cat((x0_0, self.up(x1_0)), 1))

        x2_0 = self.conv2_0(self.down(x1_0))
        x1_1 = self.conv1_1(torch.cat((x1_0, self.up(x2_0), self.down(x0_1)), 1))
        x0_2 = self.conv0_2(torch.cat((x0_1, self.up(x1_1)), 1))

        x3_0 = self.conv3_0(self.down(x2_0))
        x2_1 = self.conv2_1(torch.cat((x2_0, self.up(x3_0), self.down(x1_1)), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, self.up(x2_1), self.down(x0_2)), 1))
        x0_3 = self.conv0_3(torch.cat((x0_2, self.up(x1_2)), 1))

        x_out = self.out_conv(x0_3)
        x_out = self.hypothesis(x_out)

        return x_out