import torch
from torch.func import jvp
from tqdm import tqdm

from .composed_cpwa_net import ComposedCPWANet
from .non_linearities import (
    apply_non_linearity_and_get_activation_pattern,
    apply_non_linearity_and_update_gradient,
)
from .objects import init_knots, init_search_points
from .utilities import match_trailing_dims


def calc_remaining_dist(search_points, knots, verbose=False):
    """
    How much ground left to cover by the search points over all knots.
    """
    # first count unexplored knots
    unexplrd_idx = ~knots["started_exploring"]
    unexplrd_dist = unexplrd_idx.sum()

    # sum the distance left to cover for each search point
    srch_point_dist = (
        (search_points["max_positions"] - search_points["positions"]).clamp(min=0).sum()
    )

    total_remaining_dist = srch_point_dist + unexplrd_dist

    if verbose:
        print(
            f"search point dist: {srch_point_dist:.2f}, unexplrd dist: {unexplrd_dist:.2f}, total remaining dist: {total_remaining_dist:.2f}"
        )

    return total_remaining_dist


def t_till_next_knot(x, v, non_linearity):
    if non_linearity == "relu":
        t = -(x / v)

        t = torch.where((t <= 0) | (t.isnan()), torch.inf, t)
        return t.flatten(1).amin(1)
    elif non_linearity[:3] == "max":
        d = int(non_linearity[3])

        # find max x and indices
        x_mx, mx_idx = x.max(dim=d, keepdims=True)
        v_mx = v.gather(dim=d, index=mx_idx)

        # find intersection of lines y = x + tv with x_mx + tv_mx
        t = (x_mx - x) / (v - v_mx)
        # (0 / 0 = nan)

        t = torch.where((t <= 0) | (t.isnan()), torch.inf, t)
        return t.flatten(1).amin(1)
    elif non_linearity == "identity":
        return torch.full(size=(x.shape[0],), fill_value=torch.inf, dtype=x.dtype)
    elif non_linearity[:4] == "mean":
        return torch.full(size=(x.shape[0],), fill_value=torch.inf, dtype=x.dtype)
    elif non_linearity[:6] == "median":
        d = int(non_linearity[6])

        # find max x and indices
        x_md, md_idx = x.median(dim=d, keepdims=True)
        v_md = v.gather(dim=d, index=md_idx)

        # find intersection of lines y = x + tv with x_md + tv_md
        t = (x_md - x) / (v - v_md)
        # (0 / 0 = nan)

        t, min_step_size = match_trailing_dims(t, min_step_size)
        t = torch.where((t <= 0) | (t.isnan()), torch.inf, t)
        return t.flatten(1).amin(1)
    else:
        raise ValueError(f"non_linearity {non_linearity} not implemented")


def find_next_knot_along_direction(net: ComposedCPWANet, x, v):
    """
    Finds the time until the next knot in the direction of v.
    Both x and v are batched.
    """
    t_closest = torch.full(
        size=(x.shape[0],), fill_value=torch.inf, dtype=x.dtype, device=x.device
    )
    n = 0
    for operator, non_linearity in net.get_operators():
        n = n + 1
        # update jacobian
        x, v = jvp(operator, primals=(x,), tangents=(v,))

        # update smallest t
        t = t_till_next_knot(x, v, non_linearity)
        t_closest = torch.where(t < t_closest, t, t_closest)

        # apply non-linearity
        x, v = apply_non_linearity_and_update_gradient(x, v, non_linearity)

    return t_closest


def save_new_positions(knots, lmda, s):
    for i in range(len(s)):
        n = s[i]
        knots["positions"][n].append(lmda[i].item())


def sort_knot_positions(knots):
    # sort as if python list
    for i in range(len(knots["positions"])):
        positions = knots["positions"][i]
        knots["positions"][i] = sorted(positions)


def save_and_reallocate_search_points(search_points, knots):
    """
    Save the current search points.
    "valid" positions get saved, "invalid" positions get reallocated to new control points.

    otherwise notation becomes too verbose:
    cp => control point, sp => search point,
    alloc => allocate, realloc => reallocate
    idx => index, idc => indices,
    pos => position.
    mx => max, mn => min.
    unexplrd => unexplored.
    """

    # save the valid positions
    valid_sp_idx = search_points["positions"] < search_points["max_positions"]
    unexplrd_cp_idx = ~knots["started_exploring"]

    # if no new_idx, and no unexplored_idx, then algorithm is finished
    if (valid_sp_idx.sum() == 0) and (unexplrd_cp_idx.sum() == 0):
        return True

    # save the valid positions that weren't clamped
    save_new_positions(
        knots,
        search_points["positions"][valid_sp_idx],
        search_points["ctrl_point_idc"][valid_sp_idx],
    )

    ## allocate new search points
    # first allocate points to not yet started exploring:
    # allocate as many points as possible to unexplored
    invalid_sp_idx = ~valid_sp_idx

    # change bool indices to long indices:
    invalid_sp_idx = torch.where(invalid_sp_idx)[0]
    unexplrd_cp_idx = torch.where(unexplrd_cp_idx)[0]

    # max num sp that can be allocated to unexplored cp
    mx_alloc = min(len(invalid_sp_idx), len(unexplrd_cp_idx))

    if mx_alloc > 0:
        # perform the allocation
        search_points["ctrl_point_idc"][invalid_sp_idx[:mx_alloc]] = unexplrd_cp_idx[
            :mx_alloc
        ]
        search_points["positions"][invalid_sp_idx[:mx_alloc]] = 0
        search_points["max_positions"][invalid_sp_idx[:mx_alloc]] = 1
        knots["started_exploring"][unexplrd_cp_idx[:mx_alloc]] = True

    # allocate remaining points to already started exploring
    for realloc_sp_idx in invalid_sp_idx[mx_alloc:]:
        # TODO note: using a priority queue to track max distance to end may be more efficient,
        # but is probably overkill.
        # allocate in order of distance to end.

        dist_to_end = search_points["max_positions"] - search_points["positions"]
        midpoint = (search_points["max_positions"] + search_points["positions"]) / 2
        mx_sp_dist_idx = dist_to_end.argmax()

        # allocate the new point
        search_points["ctrl_point_idc"][realloc_sp_idx] = search_points[
            "ctrl_point_idc"
        ][mx_sp_dist_idx]
        search_points["positions"][realloc_sp_idx] = midpoint[mx_sp_dist_idx]
        search_points["max_positions"][realloc_sp_idx] = search_points["max_positions"][
            mx_sp_dist_idx
        ]

        # change the max position for the search point getting helped.
        search_points["max_positions"][mx_sp_dist_idx] = midpoint[mx_sp_dist_idx]

    return False


def update_search_points(net, search_points, knots, min_step_size=None):
    # calculate time till next knot for every point
    s = search_points["ctrl_point_idc"]

    x0 = knots["ctrl_points"][s]
    v = knots["direction_vectors"][s]
    lmda = search_points["positions"]

    eps = knots["min_step_size"][s]
    if min_step_size is not None:
        eps = eps.clamp(min=min_step_size)

    # match trailing dims for broadcasting over batch dimension
    x0, v, lmda = match_trailing_dims(x0, v, lmda)
    x = x0 + v * lmda

    # dlmda: (B,)
    dlmda = find_next_knot_along_direction(net.to(x.device), x, v)

    search_points["positions"] = lmda.view(-1) + dlmda.clamp(min=eps)


def solve_knots(
    net,
    ctrl_points,
    batch_size: int,
    max_iters: int = int(1e4),
    min_step_eps=None,
):
    knots = init_knots(ctrl_points)
    search_points = init_search_points(num_search_points=batch_size, knots=knots)

    total_dist = len(knots["ctrl_points"])
    with tqdm(total=total_dist) as pbar:
        for i in range(max_iters):
            remaining_dist = calc_remaining_dist(search_points, knots, verbose=False)
            covered_dist = total_dist - remaining_dist
            pbar.update(round(covered_dist.item() - pbar.n, 4))

            # calculate time till next knot for every point
            update_search_points(net, search_points, knots, min_step_eps)

            # save and reallocate search points
            finished = save_and_reallocate_search_points(search_points, knots)
            if finished:
                break
    pbar.close()
    sort_knot_positions(knots)

    print(f"completed at iteration {i + 1}.")
    return knots


def prune_knots(
    net,
    knots,
    batch_size: int,
):
    assert batch_size > 1
    for i, positions in enumerate(
        tqdm(knots["positions"], desc="Pruning redundant knots")
    ):
        if len(positions) == 0:
            continue

        lmbdas = torch.tensor([0] + positions + [1])  # add endpoints
        lmbdas = (lmbdas[:-1] + lmbdas[1:]) / 2  # take midpoints
        # (each midpoint should have a corresponding directional gradient)

        x0 = knots["ctrl_points"][i].unsqueeze(0)
        v = knots["direction_vectors"][i].unsqueeze(0)

        max_idx = len(lmbdas)
        prune = torch.full(size=((max_idx - 1),), fill_value=True, dtype=torch.bool)
        for begin_idx in range(0, max_idx, batch_size - 1):
            end_idx = min(max_idx, begin_idx + batch_size)
            lmda = lmbdas[begin_idx:end_idx]

            x0, v, lmda = match_trailing_dims(x0, v, lmda)
            x = x0 + lmda * v

            # check if activation patterns are the same
            prune_ = torch.full(
                size=(len(lmda) - 1,), fill_value=True, dtype=torch.bool
            )
            for operator, non_linearity in net.get_operators():
                x = operator(x)
                x, a = apply_non_linearity_and_get_activation_pattern(x, non_linearity)

                # where successive activation patterns are the same, prune the knots
                prune_ = prune_ & (
                    (~(a[:-1] == a[1:])).view(a.shape[0] - 1, -1).sum(1) == 0
                )
            prune[begin_idx : end_idx - 1] = prune[begin_idx : end_idx - 1] & prune_

        # remove the pruned lmbdas
        knots["positions"][i] = torch.tensor(positions)[~prune].tolist()
