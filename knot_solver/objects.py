import torch


def _get_minimum_detectable_step(ctrl_points):
    next, prev = ctrl_points[1:], ctrl_points[:-1]
    v = next - prev

    lowest_precision_boundary = torch.where(
        torch.abs(next) > torch.abs(prev), next, prev
    )
    min_eps = torch.finfo(v.dtype).eps * torch.abs(lowest_precision_boundary)
    interval_lens = torch.abs(v)

    ## reasoning code:
    # num_detectable_steps = interval_lens / min_eps
    # => min_detectable_step_size = 1 / num_detectable_steps
    min_detectable_step_sizes = min_eps / interval_lens

    inactive_elements = interval_lens == 0
    min_detectable_step_sizes[inactive_elements] = 0

    # heuristic: mean of min_detectable_step_sizes for active elements
    num_active_elements = (~inactive_elements).flatten(1).sum(1)
    avg_min_detectable_step_size = (
        min_detectable_step_sizes.flatten(1).sum(1) / num_active_elements
    )

    # round UP to the nearest power of 2
    rounded_avg_min_detectable_step_size = 2 ** torch.ceil(
        torch.log2(avg_min_detectable_step_size)
    )

    # finally, consider the number of values that can be stored between [0, 1], i.e how we paramterize lambda
    md_step = torch.max(
        rounded_avg_min_detectable_step_size, torch.tensor(torch.finfo(v.dtype).eps)
    )

    return md_step


def init_knots(ctrl_points):
    direction_vectors = ctrl_points[1:] - ctrl_points[:-1]
    direction_vector_lens = torch.linalg.norm(direction_vectors.flatten(1), dim=1)
    min_step_size = _get_minimum_detectable_step(ctrl_points)
    ctrl_points = ctrl_points[:-1].clone()

    knots = {
        "ctrl_points": ctrl_points,
        "positions": [[] for _ in range(len(ctrl_points))],
        "started_exploring": torch.zeros(
            len(ctrl_points), dtype=torch.bool, device=ctrl_points.device
        ),
        "direction_vectors": direction_vectors,
        "direction_vector_lens": direction_vector_lens,
        "min_step_size": min_step_size,
    }
    return knots


def init_search_points(num_search_points: int, knots, dtype=torch.float32):
    """
    Equally space search points across control points.
    """
    device = knots["ctrl_points"].device
    if num_search_points <= len(knots["ctrl_points"]):
        search_points = {
            "ctrl_point_idc": torch.arange(
                num_search_points, dtype=torch.long, device=device
            ),
            "positions": torch.zeros(
                size=(num_search_points,), dtype=dtype, device=device
            ),
            "max_positions": torch.ones(
                size=(num_search_points,), dtype=dtype, device=device
            ),
        }
        knots["started_exploring"][:num_search_points] = True
    else:
        search_points = {
            "ctrl_point_idc": torch.empty(
                num_search_points, dtype=torch.long, device=device
            ),
            "positions": torch.empty(
                size=(num_search_points,), dtype=dtype, device=device
            ),
            "max_positions": torch.empty(
                size=(num_search_points,), dtype=dtype, device=device
            ),
        }

        # equally space postions, ctrl_point_idc, and max_positions
        num_ctrl_points = len(knots["ctrl_points"])
        num_search_points_per_ctrl_point = num_search_points // num_ctrl_points
        remainder = num_search_points % num_ctrl_points

        for cp in range(num_ctrl_points):
            if cp < num_ctrl_points - remainder:
                num_allocate = num_search_points_per_ctrl_point
                begin_sp = cp * num_allocate
                end_sp = (cp + 1) * num_allocate
            else:
                num_allocate = num_search_points_per_ctrl_point + 1

                diff = cp - (num_ctrl_points - remainder)
                begin_sp_base = num_search_points_per_ctrl_point * (
                    num_ctrl_points - remainder
                )

                begin_sp = begin_sp_base + diff * num_allocate
                end_sp = begin_sp_base + (diff + 1) * num_allocate

            positions = torch.linspace(0, 1, num_allocate + 1, device=device)
            max_positions = positions[1:]
            positions = positions[:-1]

            search_points["ctrl_point_idc"][begin_sp:end_sp] = cp
            search_points["positions"][begin_sp:end_sp] = positions
            search_points["max_positions"][begin_sp:end_sp] = max_positions

        knots["started_exploring"][:] = True

    return search_points
