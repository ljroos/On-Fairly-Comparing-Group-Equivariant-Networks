def _add_trailing_dims(x, num_dims):
    return x.view(x.shape + (1,) * num_dims)


def match_trailing_dims(x, y, *args):
    # match trailing dims
    num_dims = max(x.dim(), y.dim(), *[arg.dim() for arg in args])
    x = _add_trailing_dims(x, num_dims - x.dim())
    y = _add_trailing_dims(y, num_dims - y.dim())
    args = [_add_trailing_dims(arg, num_dims - arg.dim()) for arg in args]
    return x, y, *args
