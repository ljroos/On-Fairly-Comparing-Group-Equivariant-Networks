import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torchinfo import summary

import wandb

sys.path.append("./")  # script needs to be run from main directory.

import utilities
from classifier import ComposedCPWAClassifier

from cifar_vae.interpolated_cifar import InterpolatedCIFAR
from data.image_datasets import CIFARDataModule, MNISTDataModule
from data.toy_datasets import MoonsDataModule
from data.toy_datasets import plot_batch as plot_toy_batch
from knot_solver.plot import plot_knots
from knot_solver.solver import prune_knots, solve_knots
from knot_solver.statistics import (
    calculate_directional_gradients,
    calculate_expected_gradient_norm,
    calculate_knot_entropy,
    calculate_knot_uniformity,
    calculate_max_knots_countable,
    calculate_path_len,
    calculate_smoothness,
    count_knots,
)
from mnist_vae.interpolated_mnist import InterpolatedMNIST

fig_path = utilities.get_fig_path()
save_path = utilities.get_save_path()


def main(args: argparse.Namespace):
    # seed
    if args.aa_seed is None:
        args.aa_seed = np.random.randint(int(1e7))
    pl.seed_everything(args.aa_seed)

    # logger
    if args.log_results:
        wandb_logger = pl.loggers.WandbLogger(
            entity="ljroos-msc",
            project="knot-solver-metrics",
            save_dir="logs/",
            log_model=False,  # save model locally, but not to wandb
        )
        wandb_logger.log_hyperparams(args)
    else:
        wandb_logger = False

    ## model and data
    model = ComposedCPWAClassifier(
        augment=args.augment,
        dataset=args.dataset,
        model_hparams={
            "group": args.group,
            "hidden_group_channels": args.hidden_group_channels,
            "hidden_layers": args.hidden_layers,
            "batch_norm": args.batch_norm,
            "dropout": args.dropout,
        },
        optimizer_hparams={
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "betas": (args.beta1, args.beta2),
        },
    )

    if not args.random_init_weights:
        utilities.load_model(model, args, save_path)
    model.eval()

    if args.dataset == "moons":
        dm = MoonsDataModule(
            num_canonical_samples=args.num_train,
            batch_size=args.batch_size,
            seed=args.aa_seed,
            augment=args.augment,
        )
    elif args.dataset == "mnist" or args.dataset == "downsampled_mnist":
        dm = MNISTDataModule(
            data_dir="raw-data/",
            batch_size=args.batch_size,
            num_train=args.num_train,
            num_val=0,
            num_test=10000,
            seed=args.aa_seed,
        )
    elif args.dataset == "cifar10":
        dm = CIFARDataModule(
            data_dir="raw-data/",
            batch_size=args.batch_size,
            num_train=args.num_train,
            pad_for_rotation=False,
            num_val=0,
            num_test=10000,
            seed=args.aa_seed,
        )
    else:
        raise NotImplementedError

    dm.setup(stage=None)
    dummy_x, _ = next(iter(dm.train_dataloader()))
    print(
        summary(
            model,
            model.augment(dummy_x).shape,
            device="cuda" if not args.cpu else "cpu",
        )
    )

    trainer = pl.Trainer(
        accelerator="cuda" if not args.cpu else "cpu",
        logger=wandb_logger,
    )
    trainer.validate(model, dm.train_dataloader())  # train error
    trainer.test(model, dm.test_dataloader())

    # sanity check equivariance
    if args.sanity_check_equivariance:
        model.sanity_check_equivariance(dummy_x)

    # double check that the model is in eval mode
    model.eval()
    model.to("cuda" if not args.cpu else "cpu")

    # control points stats
    num_knots = {}
    max_knots_countable = {}
    smoothness = {}
    expected_gradient_norm = {}
    knot_uniformity = {}
    knot_entropy = {}

    # control point property
    path_len = {}

    with torch.no_grad():
        if (
            args.dataset == "downsampled_mnist"
            or args.dataset == "mnist"
            or args.dataset == "cifar10"
        ):
            path = (
                (0, 8, 2, 5, 3, 7, 6, 4, 1, 9, 0)
                if args.dataset == "cifar10"
                else (0, 2, 3, 5, 8, 1, 7, 9, 4, 6, 0)
            )
            for i, j in zip(path[:-1], path[1:]):
                if args.dataset == "cifar10":
                    ctrl_points = InterpolatedCIFAR(
                        "raw-data/",
                        from_class=i,
                        to_class=j,
                        num_interpolations=args.ctrl_points_per_interpolation,
                        transform=None,  # dataset already in [-1, 1]
                    ).dataset
                else:
                    ctrl_points = InterpolatedMNIST(
                        "raw-data/",
                        from_digit=i,
                        to_digit=j,
                        num_interpolations=args.ctrl_points_per_interpolation,
                        transform=lambda x: 2 * x - 1,  # map [0, 1] to [-1, 1]
                    ).dataset

                if args.dataset == "downsampled_mnist":
                    ctrl_points = model.augment.downsample(ctrl_points)

                # solve knots after training
                knots = solve_knots(
                    net=model.model,
                    ctrl_points=ctrl_points,
                    batch_size=args.evaluate_batch_size,
                    min_step_eps=args.min_step_eps,
                )

                # prune knots
                print(f"knots before pruning: {count_knots(knots)}")
                prune_knots(
                    net=model.model,
                    knots=knots,
                    batch_size=args.evaluate_batch_size,
                )

                # calculate jacobian vector product
                calculate_directional_gradients(
                    net=model.model,
                    knots=knots,
                    batch_size=args.evaluate_batch_size,
                )

                # compute knot statistics
                num_knots[(i, j)] = count_knots(knots)
                max_knots_countable[(i, j)] = calculate_max_knots_countable(knots)
                smoothness[(i, j)] = calculate_smoothness(knots)
                expected_gradient_norm[(i, j)] = calculate_expected_gradient_norm(knots)
                knot_uniformity[(i, j)] = calculate_knot_uniformity(knots)
                knot_entropy[(i, j)] = calculate_knot_entropy(knots)

                path_len[(i, j)] = calculate_path_len(knots)

                log_dict = {
                    f"num_knots{i, j}": num_knots[(i, j)],
                    f"max_knots_countable{i, j}": max_knots_countable[(i, j)],
                    f"smoothness{i, j}": smoothness[(i, j)],
                    f"expected_gradient_norm{i, j}": expected_gradient_norm[(i, j)],
                    f"knot_uniformity{i, j}": knot_uniformity[(i, j)],
                    f"knot_entropy{i, j}": knot_entropy[(i, j)],
                }
                if args.log_results:
                    wandb.log(log_dict)

                # print log_dict
                print(f"Stats for {i} -> {j}:")
                print(20 * "*")
                for key, val in log_dict.items():
                    print(f"{key}: {val}")

                if args.plot_knots:
                    logit_path = os.path.join(fig_path, f"downsampled_logits")
                    os.makedirs(os.path.join(logit_path), exist_ok=True)
                    fig, ax = plt.subplots(1, 1)
                    plot_knots(net=model.model, knots=knots, ax=ax, which_preds=[i, j])
                    ax.set_title(
                        f"{(i, j)} {num_knots[(i, j)]} knots; {smoothness[(i, j)]:.2f} smoothness; {expected_gradient_norm[(i, j)]:.2f} EGN; {knot_uniformity[(i, j)]:.2f} unif; {knot_entropy[(i, j)]:.2f} entropy"
                    )
                    plt.savefig(os.path.join(logit_path, f"{(i, j)}logits.ignore.png"))

        elif args.dataset == "moons":
            plt_dataset = MoonsDataModule(
                num_canonical_samples=2500,
                augment="trivial",
                batch_size=1000,
            )

            ctrl_points = torch.tensor([[0, 1.5], [1, 0.5], [2, 1], [-1, 1], [0, 1.5]])

            # solve knots after training
            knots = solve_knots(
                net=model.model,
                ctrl_points=ctrl_points,
                batch_size=args.evaluate_batch_size,
                min_step_eps=args.min_step_eps,
            )

            # prune knots
            print(f"knots before pruning: {count_knots(knots)}")
            prune_knots(
                net=model.model,
                knots=knots,
                batch_size=args.evaluate_batch_size,
            )

            # calculate jacobian vector product
            calculate_directional_gradients(
                net=model.model,
                knots=knots,
                batch_size=args.evaluate_batch_size,
            )

            # compute knot statistics
            num_knots[0] = count_knots(knots)
            max_knots_countable[0] = calculate_max_knots_countable(knots)
            smoothness[0] = calculate_smoothness(knots)
            expected_gradient_norm[0] = calculate_expected_gradient_norm(knots)
            knot_uniformity[0] = calculate_knot_uniformity(knots)
            knot_entropy[0] = calculate_knot_entropy(knots)

            path_len[0] = calculate_path_len(knots)

            if args.plot_knots:
                logit_path = os.path.join(fig_path, f"moons")
                os.makedirs(os.path.join(logit_path), exist_ok=True)
                fig, ax = plt.subplots(1, 1)
                plot_knots(net=model.model, knots=knots, ax=ax)
                ax.set_title(
                    f"{num_knots[0]} knots; {smoothness[0]:.2f} smoothness; {expected_gradient_norm[0]:.2f} EGN; {knot_uniformity[0]:.2f} unif; {knot_entropy[0]:.2f} entropy"
                )
                plt.savefig(os.path.join(logit_path, "logits.ignore.png"))

            if PLOT := True:
                plt_dataset.setup(None)
                plt_loader = plt_dataset.train_dataloader()
                plt_x, plt_y = next(iter(plt_loader))

                # plot arrows between control points over the moons dataset
                fig, ax = plt.subplots(1, 1)
                plot_toy_batch(plt_x, plt_y, ax=ax, show_plot=False)
                ax.plot(
                    ctrl_points[:, 0],
                    ctrl_points[:, 1],
                    color="black",
                    linestyle="--",
                    alpha=1,
                )

                plt.savefig(os.path.join(fig_path, "moons_ctrl_points.ignore.png"))
                plt.close()

        # log knot distribution stats
        total_num_knots = 0
        total_max_knots_countable = 0
        total_knot_entropy = 0
        total_knot_uniformity = 0

        ## log fitted region stats
        total_smoothness = 0

        # expected gradient norm
        total_path_len = 0
        total_gradient_norms = 0

        for key in num_knots.keys():
            # total_path_len
            total_path_len += path_len[key].item()

            total_max_knots_countable += max_knots_countable[key]

            ## knot distribution stats
            # non-weighted stats
            total_num_knots += num_knots[key]
            total_smoothness += smoothness[key]

            # weighted avg stats
            total_gradient_norms += expected_gradient_norm[key] * path_len[key]
            total_knot_uniformity += knot_uniformity[key] * path_len[key]
            total_knot_entropy += knot_entropy[key] * path_len[key]

        expected_gradient_norm = total_gradient_norms / total_path_len
        expected_knot_uniformity = total_knot_uniformity / total_path_len
        expected_knot_entropy = total_knot_entropy / total_path_len

        log_dict = {
            "total_num_knots": total_num_knots,
            "total_max_knots_countable": total_max_knots_countable,
            "total_smoothness": total_smoothness,
            "expected_gradient_norm": expected_gradient_norm,
            "expected_knot_uniformity": expected_knot_uniformity,
            "expected_knot_entropy": expected_knot_entropy,
        }
        if args.log_results:
            wandb.log(log_dict)

        # print log_dict
        print(f"Aggregate stats:")
        print(20 * "*")
        for key, val in log_dict.items():
            print(f"{key}: {val}")


if __name__ == "__main__":
    args = utilities.parse_args(evaluater=True)
    main(args)
