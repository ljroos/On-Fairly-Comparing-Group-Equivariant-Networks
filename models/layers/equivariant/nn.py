import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .group_definitions import D4_subgroups, get_transformation_law

# For a look at how Cohen initialized equivariant layers, see: https://github.com/tscohen/GrouPy/blob/c6f40f2c07418c940e08b5297525478e3b3a824b/groupy/gconv/chainer_gconv/splitgconv2d.py, line 100.
# For in case the link breaks, go to: GrouPy -> groupy -> gconv -> chainer_gconv -> splitgconv2d.py, line 100.
# Appear they just used iid Guassians, in the same way I did it basing off of how PyTorch does it.


class D4GroupLinear(nn.Module):
    def __init__(
        self,
        in_group: str,
        out_group: str,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        if in_group != out_group and in_group != "trivial":
            raise NotImplementedError(
                f"Only in_group == out_group, or in_group='trivial' is supported for now. in_group: {in_group}, out_group: {out_group}"
            )
        self.in_group = D4_subgroups[in_group]
        self.out_group = D4_subgroups[out_group]
        self.in_features = in_features
        self.out_features = out_features

        # transformation laws
        # TODO: this will change when different in and out groups are supported.
        self.in_law, self.out_law = get_transformation_law(
            in_group, out_group, "trivial"
        )
        self.in_group_order = len(self.in_group["members"])
        self.out_group_order = len(self.out_group["members"])

        # network will have less same inference / training step time and memory usage as a regular conv layer with the same number of group expanded features.
        self.group_expanded_in_features = self.in_features * self.in_group_order
        self.group_expanded_out_features = self.out_features * self.out_group_order

        # initialize weight and bias
        self.weight = nn.Parameter(
            torch.empty(
                (
                    self.out_features,
                    self.in_features,
                    self.in_group_order,
                )
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self, init_type="default") -> None:
        if init_type == "default":
            # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
            # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
            # https://github.com/pytorch/pytorch/issues/57109
            group_channel_flattened_weight = self.weight.flatten(1, 2)
            nn.init.kaiming_uniform_(group_channel_flattened_weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    group_channel_flattened_weight
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            raise NotImplementedError(
                "Other initialization methods are not implemented yet."
            )

    def group_expand_weight(self) -> torch.Tensor:
        # TODO: this needs to change when different in and out groups are supported.
        group_expanded_weight = torch.stack(
            [self.in_law(self.weight, g) for g in range(self.out_group_order)],
            dim=1,
        )
        return group_expanded_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_expanded_weight = self.group_expand_weight().view(
            self.group_expanded_out_features,
            self.group_expanded_in_features,
        )
        out = F.linear(x.flatten(-2), group_expanded_weight, None)
        out = out.view(*x.shape[:-2], self.out_features, self.out_group_order)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(-1)
        return out

    def extra_repr(self) -> str:
        return "in_features={}, in_group={}, out_features={}, out_group={}, bias={}".format(
            self.in_features,
            self.in_group["name"],
            self.out_features,
            self.out_group["name"],
            self.bias is not None,
        )


class D4GroupConv2d(nn.Module):
    def __init__(
        self,
        in_group: str,
        out_group: str,
        in_channels: int,
        out_channels: int,
        # ks, stride, etc. forced to be square.
        kernel_size: int,
        stride: int = 1,
        # assume padding mode is "zeros".
        padding: int = 0,
        dilation: int = 1,
        num_channel_groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_group != out_group and in_group != "trivial":
            raise NotImplementedError(
                f"Only in_group == out_group, or in_group='trivial' is supported for now. in_group: {in_group}, out_group: {out_group}"
            )
        self.in_group = D4_subgroups[in_group]
        self.out_group = D4_subgroups[out_group]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.num_channel_groups = num_channel_groups  # nothing to do with group theory. num_groups parameter of regular conv layer.

        # transformation laws
        # TODO: this will change when different in and out groups are supported.
        self.in_law, self.out_law = get_transformation_law(in_group, out_group, "z2")
        self.in_group_order = len(self.in_group["members"])
        self.out_group_order = len(self.out_group["members"])

        # network will have less same inference / training step time and memory usage as a regular conv layer with the same number of group expanded channels.
        self.group_expanded_in_channels = self.in_channels * self.in_group_order
        self.group_expanded_out_channels = self.out_channels * self.out_group_order

        # initialize weight and bias
        self.weight = nn.Parameter(
            torch.empty(
                (
                    self.out_channels,
                    self.in_channels // self.num_channel_groups,
                    self.in_group_order,
                    *self.kernel_size,
                )
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self, init_type="default") -> None:
        if init_type == "default":
            # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
            # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
            # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
            group_channel_flattened_weight = self.weight.flatten(1, 2)
            nn.init.kaiming_uniform_(group_channel_flattened_weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    group_channel_flattened_weight
                )
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(self.bias, -bound, bound)
        else:
            raise NotImplementedError(
                "Other initialization methods are not implemented yet."
            )

    def group_expand_weight(self) -> torch.Tensor:
        # TODO: this needs to change when different in and out groups are supported.
        group_expanded_weight = torch.stack(
            [self.in_law(self.weight, g) for g in range(self.out_group_order)],
            dim=1,
        )
        return group_expanded_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_expanded_weight = self.group_expand_weight().view(
            self.group_expanded_out_channels,
            self.group_expanded_in_channels,
            *self.kernel_size,
        )
        out = F.conv2d(
            input=x.flatten(1, 2),
            weight=group_expanded_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.num_channel_groups,
        )
        out = out.view(
            x.shape[0], self.out_channels, self.out_group_order, *out.shape[-2:]
        )
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out


class MaxPool2d(nn.Module):
    "Flattens group dimension, then performs maxpooling, then reshapes back to original shape."

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.maxpool = nn.MaxPool2d(*args, **kwargs)

    def forward(self, x):
        # x shape = (batch, group, channel, height, width)
        # reshape to (B, G * C, H, W)
        shape = x.shape
        x = x.view(shape[0], -1, *shape[3:])
        # max may break rot equivariance if not done carefully.
        x = self.maxpool(x)
        x = x.view(*shape[:3], *x.shape[-2:])
        return x


class AdaptiveAvgPool2d(nn.Module):
    "Flattens group dimension, then performs adaptive average pooling, then reshapes back to original shape."

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(*args, **kwargs)

    def forward(self, x):
        # x shape = (batch, group, channel, height, width)
        # reshape to (B, G * C, H, W)
        shape = x.shape
        x = x.view(shape[0], -1, *shape[3:])
        # max may break rot equivariance if not done carefully.
        x = self.avgpool(x)
        x = x.view(*shape[:3], *x.shape[-2:])
        return x
