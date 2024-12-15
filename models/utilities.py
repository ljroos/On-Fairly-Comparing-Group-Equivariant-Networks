import torch.nn as nn


def debug_print(x):
    print(x)
    return x


act_fn_dict = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,
}
