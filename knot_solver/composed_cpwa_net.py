from abc import abstractmethod

import torch
import torch.nn as nn
from torch.func import jvp

from .non_linearities import (
    apply_non_linearity,
    apply_non_linearity_and_get_activation_pattern,
    apply_non_linearity_and_update_gradient,
)


class ComposedCPWANet(nn.Module):
    """
    Composed Continuous Piecewise Affine Network.
    """

    @abstractmethod
    def get_operators(self):
        """
        Returns an iterable of tuples of the form (operator, non_linearity).
        """
        raise NotImplementedError

    def forward(self, x):
        operators = self.get_operators()
        for operator, non_linearity in operators:
            x = operator(x)
            x = apply_non_linearity(x, non_linearity)
        return x

    def act_pattern(self, x):
        pattern = []
        for operator, non_linearity in self.get_operators():
            x = operator(x)
            x, a = apply_non_linearity_and_get_activation_pattern(x, non_linearity)
            pattern.append(a.flatten(1))
        pattern = torch.cat(pattern, dim=1)
        return pattern

    def directional_gradient(self, x, v):
        """
        Returns the directional gradient of the network at x in the direction of v.
        """
        operators = self.get_operators()
        for operator, non_linearity in operators:
            x, v = jvp(operator, primals=(x,), tangents=(v,))
            x, v = apply_non_linearity_and_update_gradient(x, v, non_linearity)
        return v
