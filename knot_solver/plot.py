import matplotlib.pyplot as plt
import numpy as np
import torch

from .statistics import count_knots
from .utilities import match_trailing_dims


def print_all_knots(knots, max_print=10):
    N = len(knots["ctrl_points"])
    for n in range(N):
        if n >= max_print:
            print("...")
            break
        positions = knots["positions"][n]
        print(f"ctrl p {n}: ", sep="", end="")
        for position in positions:
            print(f"{position:.2f}", end=", ")
        print()


def plot_knot_distribution(knots):
    knots = []
    for positions in knots["positions"]:
        knots.append(len(positions))
    knots = np.array(knots)

    _, ax = plt.subplots(1, 2)

    # plot knot counts
    ax[0].bar(np.arange(len(knots), dtype=int), knots)
    ax[0].title(f"total knots: {knots.sum()}")
    ax[0].xticks([])
    ax[0].xlabel("control point")
    ax[0].ylabel("num knots")

    # plot knots per length of control
    ax[1].bar(np.arange(len(knots), dtype=int), knots)
    ax[1].title(f"total knots: {knots.sum()}")
    ax[1].xticks([])
    ax[1].xlabel("control point")
    ax[1].ylabel("knots / len")

    plt.show()


# plot knots
def plot_knots(net, knots, ax, which_preds=None):
    with torch.no_grad():
        v = knots["direction_vectors"]
        dists = torch.linalg.norm(v.flatten(1), dim=1)
        cum_dists = dists.cumsum(dim=0)
        total_dist = cum_dists[-1]
        cum_dists = torch.cat([torch.zeros((1,), device=v.device), cum_dists])

        # plot a vertical dotted line at every control point
        # for i in range(len(knots["ctrl_points"])):
        #     plt.axvline(cum_dists[i].item(), color="gray", linestyle="--", alpha=0.25)

        for i in range(len(knots["positions"])):
            # TODO: this gray dotted line does not work if using
            # normalized_v setting in solve_knots
            lmda = torch.linspace(0, 1, 1000, device=v.device)
            knot_dists = lmda * dists[i] + cum_dists[i]

            x0 = knots["ctrl_points"][[i]]
            v0 = knots["direction_vectors"][[i]]

            x0, v0, lmda = match_trailing_dims(x0, v0, lmda)
            x = x0 + v0 * lmda
            preds = net(x).view(x.shape[0], -1)

            knot_dists = knot_dists.detach().cpu()
            preds = preds.detach().cpu()

            # only plot max pred
            # which_preds = preds.argmax(dim=1, keepdims=True)
            if which_preds is None:
                which_preds = (0,)

            for p, col in zip(which_preds, ["gray", "black", "brown"]):
                # grayscale colors:
                ax.plot(
                    knot_dists,
                    preds[:, p],
                    label=f"curve {p}",
                    alpha=0.25,
                    c=col,
                )

            if len(knots["positions"][i]) == 0:
                continue

            lmda = torch.tensor(knots["positions"][i], device=v.device)
            knot_dists = lmda * dists[i] + cum_dists[i]

            x0 = knots["ctrl_points"][[i]]
            v0 = knots["direction_vectors"][[i]]

            x0, v0, lmda = match_trailing_dims(x0, v0, lmda)
            x = x0 + v0 * lmda
            preds = net(x).view(x.shape[0], -1)

            knot_dists = knot_dists.detach().cpu()
            preds = preds.detach().cpu()

            for p, _ in zip(which_preds, ["gray", "black", "brown"]):
                ax.scatter(knot_dists, preds[:, p], label=f"knot {i}", alpha=0.25)

    total_knots = count_knots(knots)
    ax.set_xlim(0, total_dist.item())
    ax.set_xticks([])
    ax.set_title(f"total knots: {total_knots}")
