import sys

import torch
import torch.nn as nn

sys.path.append("./")


from knot_solver.composed_cpwa_net import ComposedCPWANet
from models.layers.equivariant import nn as enn
from models.layers.equivariant.group_definitions import D4_subgroups


class GroupLift(nn.Module):
    def __init__(self, group: str) -> None:
        super().__init__()
        self.group = group
        self.flipW = nn.Parameter(torch.tensor([-1.0, 1.0]), requires_grad=False)
        self.flipH = nn.Parameter(torch.tensor([1.0, -1.0]), requires_grad=False)
        self.flipHW = nn.Parameter(torch.tensor([-1.0, -1.0]), requires_grad=False)

    def forward(self, x):
        X = [x]
        if self.group == "trivial":
            pass
        elif self.group == "flipH":
            X.append(x * self.flipH)
        elif self.group == "flipW":
            X.append(x * self.flipW)
        elif self.group == "rot180":
            X.append(x * self.flipHW)
        elif self.group == "flipH_and_or_flipW":
            X.append(x * self.flipH)
            X.append(x * self.flipW)
            X.append(x * self.flipHW)
        else:
            raise NotImplementedError(f"group {self.group} not implemented.")
        x = torch.stack(X, dim=2)
        return x


class MoonsClassifier(ComposedCPWANet):
    def __init__(
        self,
        group: str,
        hidden_group_channels: int,
        hidden_layers: int,
        batch_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_group_channels = hidden_group_channels
        self.group = group
        self.group_size = len(D4_subgroups[group]["members"])
        assert (
            hidden_group_channels % self.group_size == 0
        ), "hidden_group_channels must be divisible by group_size"
        self.hidden_channels = self.hidden_group_channels // self.group_size

        self.lift = GroupLift(group)
        self.initial_conv = enn.D4GroupLinear(
            in_group=group,
            out_group=group,
            in_features=2,
            out_features=self.hidden_channels,
            bias=not batch_norm,
        )
        self.initial_bn = (
            nn.BatchNorm1d(self.hidden_channels) if batch_norm else nn.Identity()
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        assert hidden_layers >= 1
        for _ in range(hidden_layers - 1):
            self.convs.append(
                enn.D4GroupLinear(
                    in_group=group,
                    out_group=group,
                    in_features=self.hidden_channels,
                    out_features=self.hidden_channels,
                    bias=not batch_norm,
                )
            )
            self.norms.append(
                nn.BatchNorm1d(self.hidden_channels) if batch_norm else nn.Identity()
            )

        self.dropout = nn.Dropout1d(dropout)

        self.out_conv = enn.D4GroupLinear(
            in_group=group,
            out_group=group,
            in_features=self.hidden_channels,
            out_features=4 // self.group_size,
            bias=True,
        )

    def in_operator(self):
        def operator(x):
            x = self.lift(x)
            x = self.initial_conv(x)
            x = self.initial_bn(x)
            x = self.dropout(x)
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
            x = x.view(x.shape[0], 4)
            return x

        return operator, "max1"

    def get_operators(self):
        operators = [self.in_operator()]
        for i in range(len(self.convs)):
            operators.append(self.hidden_operator(i))
        operators.append(self.out_operator())
        return operators
