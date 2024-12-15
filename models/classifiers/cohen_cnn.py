"""
Cohen and Welling's GCNN for MNIST classification, adapted from their repo:
https://github.com/tscohen/gconv_experiments/blob/master/gconv_experiments/MNIST_ROT/models/P4CNN.py

Uses 'num_group_expanded_hidden channels' hidden channels, which is = hidden_channels * group_size.
Forces that any two models with the same number of group_expanded hidden channels will
have the same computational cost in a forward and backward pass in terms of number of operations, regardless of the selected group.
"""

import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.append("./")

from knot_solver.composed_cpwa_net import ComposedCPWANet
from models.layers.equivariant import nn as genn
from models.layers.equivariant.group_definitions import D4_subgroups


class ConvBN(nn.Module):
    """
    In our experiments we let a different class handle the activation function.
    Format kept for comparing with Cohen's original code.
    """

    def __init__(self, conv: nn.Module, bn: bool = True):
        super().__init__()
        self.conv = conv
        self.bn = (
            nn.BatchNorm3d(conv.out_channels) if bn else nn.Identity()
        )  # use 3d batchnorm to support equivariance

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class P4MCNN(ComposedCPWANet):
    """
    p4m - group of 90 degree rotations and reflections + translations
    """

    def __init__(
        self,
        group: str,
        hidden_group_channels: int,
        # default values, used from Cohen
        hidden_layers: int = 6,
        dropout_p: float = 0.0,
        batch_norm: bool = True,
        kernel_size: int = 3,
        in_channels: int = 1,
    ):
        super().__init__()
        self.group = group
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.group_size = len(D4_subgroups[group]["members"])

        self.hidden_layers = hidden_layers

        # controls for amount of compute
        assert (
            hidden_group_channels % self.group_size == 0
        ), "hidden_group_channels must be divisible by group_size"
        self.hidden_channels = hidden_group_channels // self.group_size

        # corresponds to l1 in the repo
        # lift from trivial group to group
        self.bot = ConvBN(
            conv=genn.D4GroupConv2d(
                in_group="trivial",
                out_group=group,
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=not batch_norm,
            ),
            bn=batch_norm,
        )

        # corresponds to l2-l6 in the repo
        self.conv_layers = nn.ModuleList()
        for _ in range(self.hidden_layers - 1):
            self.conv_layers.append(
                ConvBN(
                    conv=genn.D4GroupConv2d(
                        in_group=group,
                        out_group=group,
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=0,
                        bias=not batch_norm,
                    ),
                    bn=self.batch_norm,
                )
            )

        # when group is D4, this is 10.
        # Used to compare with other groups so that compute is matched.
        out_channels = 80 // self.group_size

        # corresponds to top in the repo
        self.top = genn.D4GroupConv2d(
            in_group=group,
            out_group=group,
            in_channels=self.hidden_channels,
            kernel_size=4,
            out_channels=out_channels,
            bias=True,
        )

        self.drop = nn.Dropout3d(p=dropout_p)  # use 3d dropout to support equivariance

    def in_operator(self):
        def operator(x):
            x = x.unsqueeze(2)  # add group dimension
            x = self.bot(x)
            x = self.drop(x)

            return x

        return operator, "relu"

    @staticmethod
    def _spatial_pooling(x):
        # store initial shape and flatten
        initial_shape = x.shape
        x = x.view(initial_shape[0], -1, *initial_shape[3:])

        # max may break rot equivariance if not done carefully.
        x = F.avg_pool2d(
            x, kernel_size=2, stride=2, padding=0, ceil_mode=True
        )  # for simplicity for solving knots, we use avg pooling instead of max pooling (mean is linear, max is not)

        # reshape back to original shape
        x = x.view(*initial_shape[:3], *x.shape[-2:])

        return x

    def hidden_operator(self, i):
        def operator(x):
            if i == 1:
                x = self._spatial_pooling(x)

            x = self.conv_layers[i](x)
            x = self.drop(x)
            # original code doesn't have dropout at last conv, but we add for consistency

            return x

        return operator, "relu"

    def out_operator(self):
        def operator(x):
            x = self.top(x)

            # x shape = (batch, channels=10, D4_group_size=8, height, width)
            x = x.view(x.shape[0], 10, 8, *x.shape[-2:])
            x = x.flatten(start_dim=2)

            return x

        return operator, "max2"

    def get_operators(self):
        operators = [self.in_operator()]
        for i in range(len(self.conv_layers)):
            operators.append(self.hidden_operator(i))
        operators.append(self.out_operator())
        return operators
