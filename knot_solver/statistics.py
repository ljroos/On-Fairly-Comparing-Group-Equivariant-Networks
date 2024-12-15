import numpy as np
import torch
from tqdm import tqdm, trange

from .composed_cpwa_net import ComposedCPWANet
from .utilities import match_trailing_dims


def count_knots(knots):
    return np.sum([len(positions) for positions in knots["positions"]])


def calculate_path_len(knots):
    return knots["direction_vector_lens"].sum()


def calculate_max_knots_countable(knots):
    min_step_size = knots["min_step_size"]
    return (1 / min_step_size).floor().sum().item()


def calculate_directional_gradients(net: ComposedCPWANet, knots, batch_size: int):
    """
    knots["positions"] is a list of lists of positions.
    This function calculates the directional gradient for each position,
    including between end points 0 and 1, and stores it in a corresponding list
    of vectors.
    """
    assert batch_size > 1
    knots["directional_gradients"] = []
    v_lens = knots["direction_vector_lens"]
    for i, positions in enumerate(
        tqdm(knots["positions"], desc="calculating jacobian vector products")
    ):
        lmbdas = torch.tensor([0] + positions + [1])  # add endpoints

        x0 = knots["ctrl_points"][i].unsqueeze(0)
        v = knots["direction_vectors"][i].unsqueeze(0)

        max_idx = len(lmbdas)
        directional_gradients = []
        for begin_idx in range(0, max_idx, batch_size - 1):
            end_idx = min(max_idx, begin_idx + batch_size)
            lmda = lmbdas[begin_idx:end_idx]
            dx = (lmda[1:] - lmda[:-1]) * v_lens[i]

            x0, v, lmda = match_trailing_dims(x0, v, lmda)
            x = x0 + lmda * v

            out = net(x)
            dy = out[1:] - out[:-1]

            dx, dy = match_trailing_dims(dx, dy)
            directional_gradient = dy / dx
            directional_gradients.append(directional_gradient)

        directional_gradients = torch.cat(directional_gradients, dim=0)
        assert len(directional_gradients) == len(positions) + 1
        knots["directional_gradients"].append(directional_gradients)


def calculate_smoothness(knots):
    smoothness = 0
    for i in trange(len(knots["directional_gradients"]), desc="calculating smoothness"):
        grads = knots["directional_gradients"][i]

        assert len(grads) == len(knots["positions"][i]) + 1

        smoothness += ((grads[1:] - grads[:-1]) ** 2).sum().item()

        if i + 1 < len(knots["directional_gradients"]):
            next_grad = knots["directional_gradients"][i + 1][0]
            smoothness += ((next_grad - grads[-1]) ** 2).sum().item()
    return smoothness


def calculate_expected_gradient_norm(knots):
    v_lens = knots["direction_vector_lens"]
    total_len = v_lens.sum()

    weighted_grad_norm_sum = 0
    for i in trange(
        len(knots["directional_gradients"]), desc="calculating directional gradients"
    ):
        # weight for every direction norm
        lmbdas = torch.tensor([0] + knots["positions"][i] + [1])
        weights = (lmbdas[1:] - lmbdas[:-1]) * v_lens[i]

        # calculate directional gradient norms
        grads = knots["directional_gradients"][i]
        grads_norm = torch.linalg.norm(grads.view(grads.shape[0], -1), dim=1)

        weighted_grad_norm_sum += (grads_norm * weights).sum().item()

    return weighted_grad_norm_sum / total_len


def get_absolute_knot_positions(knots):
    """
    Returns one long array with all the knot positions in absolute terms over the curve.
    """
    v_lens = knots["direction_vector_lens"]
    cum_positions = torch.cumsum(v_lens, dim=0)
    cum_positions = torch.cat(
        [torch.tensor([0.0], device=cum_positions.device), cum_positions], dim=0
    )

    absolute_knot_positions = []
    k = 0
    for i in range(len(knots["positions"])):
        lmbdas = torch.tensor(knots["positions"][i])
        if len(lmbdas) == 0:
            continue
        absolute_knot_positions += (cum_positions[i] + lmbdas * v_lens[i]).tolist()

        l = len(lmbdas)
        k = k + l

    return absolute_knot_positions


def calculate_knot_uniformity(knots, midpoint_approximation=True):
    """
    The knot uniformity metric omega^2. Matches notation from paper.
    """
    x = get_absolute_knot_positions(knots)
    L = calculate_path_len(knots)

    # include endpoints
    x.insert(0, 0)
    x.append(L.item())
    x = torch.tensor(x, dtype=torch.float64)

    # define knot distances
    d1 = x[1:] - x[:-1]
    d2 = x[1:] ** 2 - x[:-1] ** 2
    d3 = x[1:] ** 3 - x[:-1] ** 3

    # define lengths
    N = len(x) - 2  # x includes 0 and L
    M = 1 / (N + 1)
    m = M / d1

    # define ECDF at knot points
    F = torch.arange(N + 2) / (N + 1)

    if midpoint_approximation:
        # alternative C calculation:
        x_m = (x[:-1] + x[1:]) / 2
        C = (F[:-1] + M / 2 - x_m / L) ** 2 * d1
    else:
        # alert with warning
        print("WARNING; calculating C without midpoint approximation is unstable.")

        A = F[:-1] - m * x[:-1]
        B = m - (1 / L)

        T1 = (A**2) * d1
        T2 = A * B * d2
        T3 = (B**2) * d3 / 3

        C = T1 + T2 + T3

    return C.sum() / L


def calculate_knot_entropy(knots):
    x = get_absolute_knot_positions(knots)
    L = calculate_path_len(knots)

    # include endpoints
    x.insert(0, 0)
    x.append(L.item())
    x = torch.tensor(x, dtype=torch.float64)

    # define knot distances
    d1 = x[1:] - x[:-1]

    # define lengths
    N = len(x) - 2  # x includes 0 and L

    I = torch.log(d1)

    # If I contains infs
    if torch.isinf(I).any():
        print(
            "WARNING; I contains infs. Removing before taking mean, scaling up remaining values."
        )
        # replace infs with min eps
        # this is a hack to avoid infs in the log
        min_allowed_log_value = torch.log(torch.tensor(torch.finfo(torch.float64).eps))
        I = I.clamp(min=min_allowed_log_value)

    return np.log(N + 1) + I.sum() / (N + 1)
