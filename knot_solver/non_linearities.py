import torch


def apply_non_linearity(x, non_linearity: str):
    if non_linearity == "relu":
        return torch.relu(x)
    elif non_linearity[:3] == "max":
        return torch.amax(x, dim=int(non_linearity[3]))
    elif non_linearity[:4] == "mean":
        return x.mean(dim=int(non_linearity[4]))
    elif non_linearity[:6] == "median":
        return x.median(dim=int(non_linearity[6]))[0]
    elif non_linearity == "identity":
        return x
    else:
        raise ValueError(f"Unknown non-linearity: {non_linearity}")


def apply_non_linearity_and_get_activation_pattern(x, non_linearity: str):
    if non_linearity == "relu":
        return torch.relu(x), (x > 0).int()
    elif non_linearity[:3] == "max":
        x_mx, mx_idx = x.max(dim=int(non_linearity[3]))
        return x_mx, mx_idx
    elif non_linearity[:4] == "mean":
        return x.mean(dim=int(non_linearity[4])), torch.full_like(x, 0, dtype=torch.int)
    elif non_linearity[:6] == "median":
        x_md, md_idx = x.median(dim=int(non_linearity[6]))
        return x_md, md_idx
    elif non_linearity == "identity":
        return x, torch.full_like(x, 0, dtype=torch.int)
    else:
        raise ValueError(f"Unknown non-linearity: {non_linearity}")


def apply_non_linearity_and_update_gradient(x, v, non_linearity: str):
    """
    WARNING: Using max or median is only truly supported for final layers.

    To use max or median for intermediate layers, you need to implement the derivative of the max or median,
    but when there is a tie breaking, the derivative is not well defined, and the directional derivative is hard to implement.
    """
    if non_linearity == "relu":
        # default PyTorch does not take into account directional derivative of ReLU
        # if moving towards right, then derivative at 0 is 1
        v = torch.where((v > 0) & (x == 0), v, v * (x > 0).float())
        return torch.relu(x), v
    elif non_linearity[:3] == "max":
        d = int(non_linearity[3])

        x_mx, mx_idx = x.max(dim=d, keepdim=True)
        v_mx = v.gather(dim=d, index=mx_idx)
        return x_mx.squeeze(d), v_mx.squeeze(d)
    elif non_linearity[:4] == "mean":
        d = int(non_linearity[4])
        return x.mean(dim=d), v
    elif non_linearity[:6] == "median":
        d = int(non_linearity[6])

        x_md, md_idx = x.median(dim=d)
        v_md = v.gather(dim=d, index=md_idx, keepdim=True)
        return x_md.squeeze(d), v_md.squeeze(d)
    elif non_linearity == "identity":
        return x, v
    else:
        raise ValueError(f"Unknown non-linearity: {non_linearity}")
