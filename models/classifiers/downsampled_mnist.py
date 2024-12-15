import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("./")

from knot_solver.composed_cpwa_net import ComposedCPWANet


def _create_perm_matrix(square_side_len):
    """
    Utility for implementing TranslateH_and_W group (i.e. diagonal translation).
    """
    perm_matrix = []
    for K in range(square_side_len):
        k = K
        p = [k]
        while True:
            h = k // square_side_len
            w = k % square_side_len
            h_ = (h + 1) % square_side_len
            w_ = (w + 1) % square_side_len
            k = h_ * square_side_len + w_
            if k == K:
                break
            p.append(k)
        perm_matrix.append(p)
    return torch.tensor(perm_matrix)


class WrappedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        # Initialize the standard Conv2d layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        # Calculate the padding needed to mimic wrapping based on the kernel size
        # Adjust padding based on the dilation
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        # Apply circular padding to the input
        x_padded = F.pad(x, (self.padding, self.padding), mode="circular")
        # Apply the convolution to the padded input
        return self.conv(x_padded)


class WrappedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        # Initialize the standard Conv2d layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        # Calculate the padding needed to mimic wrapping based on the kernel size
        # Adjust padding based on the dilation
        self.padding = [(k - 1) // 2 for k in self._pair(kernel_size)]

    def forward(self, x):
        # Apply circular padding to the input
        x_padded = F.pad(
            x,
            (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
            mode="circular",
        )
        # Apply the convolution to the padded input
        return self.conv(x_padded)

    def _pair(self, x):
        # A utility method to ensure kernel_size, stride, etc., are tuples
        if isinstance(x, int):
            return (x, x)
        return x


class DownsampledMNISTClassifier(ComposedCPWANet):
    def __init__(
        self,
        group: str,
        hidden_group_channels: int,
        hidden_layers: int,
        batch_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.group = group
        if self.group == "trivial":
            self.group_size = 1
            Conv = lambda in_channels, hidden_channels, bias: nn.Linear(
                in_channels, hidden_channels, bias=bias
            )
            Norm = nn.BatchNorm1d
            Drop = nn.Dropout
            self.initial_reshape = lambda x: x.flatten(1)
            self.initial_channels = 49
        elif self.group in ["translateH", "translateW", "translateH_and_W"]:
            self.group_size = 7
            Conv = lambda in_channels, hidden_channels, bias: WrappedConv1d(
                in_channels, hidden_channels, 7, bias=bias
            )
            Norm = nn.BatchNorm1d
            Drop = nn.Dropout1d
            if self.group == "translateH":
                self.initial_reshape = lambda x: x.flatten(1, 2)
            elif self.group == "translateW":
                self.initial_reshape = lambda x: x.transpose(2, 3).flatten(1, 2)
            else:
                # case translateH_and_W
                # more involved reshaping; permute elements such that those on the same diagonal are in the same row.
                # Then we reuse the same techniques as translateH.
                pixel_mapping = _create_perm_matrix(7)
                pixel_permute = lambda x: x.flatten(start_dim=2)[
                    :, :, pixel_mapping.flatten()
                ].view(x.shape[0], x.shape[1], 7, 7)
                self.initial_reshape = lambda x: pixel_permute(x).flatten(1, 2)
            self.initial_channels = 7
        elif self.group == "translateH_and_or_W":
            self.group_size = 49
            Conv = lambda in_channels, hidden_channels, bias: WrappedConv2d(
                in_channels, hidden_channels, 7, bias=bias
            )
            Norm = nn.BatchNorm2d
            Drop = nn.Dropout2d
            self.initial_reshape = lambda x: x
            self.initial_channels = 1
        else:
            raise ValueError(f"Unknown group {group}")
        if hidden_group_channels % self.group_size != 0:
            print(
                "Warning, hidden_group_channels is not divisible by group_size. Rounding down."
            )
        self.hidden_channels = hidden_group_channels // self.group_size
        self.hidden_group_channels = self.hidden_channels * self.group_size

        # assumes input image size is 7x7
        self.initial = nn.Sequential(
            Conv(self.initial_channels, self.hidden_channels, bias=not batch_norm),
            nn.Identity() if batch_norm else Norm(self.hidden_channels),
            Drop(dropout),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        assert hidden_layers >= 1
        for _ in range(hidden_layers - 1):
            self.convs.append(
                Conv(
                    self.hidden_channels,
                    self.hidden_channels,
                    bias=not batch_norm,
                )
            )
            self.norms.append(Norm(self.hidden_channels))
        self.dropout = Drop(dropout)

        self.out_conv = Conv(
            self.hidden_channels, 10 * (49 // self.group_size), bias=True
        )

    def in_operator(self):
        def operator(x):
            x = self.initial_reshape(x)

            # initial conv
            x = self.dropout(self.initial(x))

            return x

        return operator, "relu"

    def hidden_operator(self, i):
        def operator(x):
            x = self.convs[i](x)
            x = self.norms[i](x)
            x = self.dropout(x)

            return x

        return operator, "relu"

    def out_operator(self):
        def operator(x):
            x = self.out_conv(x)
            x = x.view(x.shape[0], 10, 49)

            return x

        return operator, "max2"

    def get_operators(self):
        operators = [self.in_operator()]
        for i in range(len(self.convs)):
            operators.append(self.hidden_operator(i))
        operators.append(self.out_operator())
        return operators
