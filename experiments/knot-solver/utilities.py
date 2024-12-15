"""
Module for loading arguments for trainer and evaluater.
"""

import argparse
import os
import sys

import pytorch_lightning as pl
import torch

sys.path.append("./")  # script needs to be run from main directory.

SEPERATOR = "+"


def get_fig_path() -> str:
    path = os.path.join(
        "experiments",
        "knot-solver",
        "figures",
    )
    os.makedirs(path, exist_ok=True)
    return path


def get_save_path() -> str:
    path = os.path.join(
        "ljroos-msc-model-weights",
        "knot-solver",
    )
    os.makedirs(path, exist_ok=True)
    return path


def args_to_save_name(args: argparse.Namespace) -> str:
    """
    Returns the path to save the model weights to.
    """
    save_name = (
        args.group
        + SEPERATOR
        + str(args.hidden_group_channels)
        + SEPERATOR
        + str(args.hidden_layers)
        + SEPERATOR
        + str(args.batch_norm)
        + SEPERATOR
        + str(args.dropout)
        + SEPERATOR
        + str(args.learning_rate)
        + SEPERATOR
        + str(args.weight_decay)
        + SEPERATOR
        + str(args.beta1)
        + SEPERATOR
        + str(args.beta2)
        + SEPERATOR
        + str(args.batch_size)
        + SEPERATOR
        + str(args.max_epochs)
        + SEPERATOR
        + str(args.patience)
        + SEPERATOR
        + args.dataset
        + SEPERATOR
        + args.augment
        + SEPERATOR
        + str(args.num_train)
        + SEPERATOR
        + str(args.aa_seed)
        + ".pt"
    )
    return save_name


def save_name_to_args(save_name: str) -> argparse.Namespace:
    """
    Returns the arguments used to save the model weights with save_name.
    """
    save_name = save_name.split(".pt")[0]
    save_name = save_name.split(SEPERATOR)
    args = argparse.Namespace()
    args.group = save_name[0]
    args.hidden_group_channels = int(save_name[1])
    args.hidden_layers = int(save_name[2])
    args.batch_norm = bool(save_name[3])
    args.dropout = float(save_name[4])
    args.learning_rate = float(save_name[5])
    args.weight_decay = float(save_name[6])
    args.beta1 = float(save_name[7])
    args.beta2 = float(save_name[8])
    args.batch_size = int(save_name[9])
    args.max_epochs = int(save_name[10])
    args.patience = int(save_name[11])
    args.dataset = save_name[12]
    args.augment = save_name[13]
    args.num_train = int(save_name[14])
    args.aa_seed = int(save_name[15])
    return args


def save_model(model: pl.LightningModule, args: argparse.Namespace, save_path: str):
    """
    Stores model weights to disk at the location defined by args.
    """
    save_name = args_to_save_name(args)
    save_loc = os.path.join(save_path, save_name)
    torch.save(model.cpu().state_dict(), save_loc)


def load_model(model: pl.LightningModule, args: argparse.Namespace, save_path: str):
    """
    Loads model weights from disk at the location defined by args.
    """
    save_name = args_to_save_name(args)
    save_loc = os.path.join(save_path, save_name)
    model.load_state_dict(torch.load(save_loc))


def parse_args(evaluater: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # net settings
    parser.add_argument(
        "--group",
        type=str,
        default="trivial",
        required=False,
        help=f"which group to use for equivariance.",
    )
    parser.add_argument(
        "--hidden_group_channels",
        default=16,
        type=int,
        help="Number of channels for the group hidden layer.",
        required=False,
    )
    parser.add_argument(
        "--hidden_layers",
        default=1,
        type=int,
        help="number of hidden layers.",
        required=False,
    )
    parser.add_argument(
        "--batch_norm",
        action="store_true",
        default=False,
        help="Enable batch norm for convolutional layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate",
    )

    # optimizer (AdamW)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Model learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="AdamW beta1.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="AdamW beta2.",
    )

    # training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Max number of epochs to train for.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Patience for early stopping.",
    )

    # dataset / task
    parser.add_argument(
        "--dataset",
        default="moons",
        choices=["moons", "mnist", "downsampled_mnist", "cifar10"],
        help="Different toy datasets",
    )
    parser.add_argument(
        "--augment",
        type=str,
        default="trivial",
        required=False,
        help=f"which group to use to augment dataset",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=60000,
        help="Number of training samples (in canonical representation).",
    )

    # miscellaneous
    parser.add_argument(
        "--aa_seed",
        type=int,
        default=0,
        help="Random seed to use. If < 0, uses a random seed.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Run on CPU instead of GPU.",
    )
    parser.add_argument(
        "--log_results",
        action="store_true",
        default=False,
        help="log results to wandb",
    )
    parser.add_argument(
        "--sanity_check_equivariance",
        action="store_true",
        default=False,
        help="Check equivariance of model on a batch of data.",
    )

    if evaluater:
        # knot solver
        parser.add_argument(
            "--min_step_eps",
            type=float,
            default=1e-6,
            help="Min step size for knot solver.",
        )
        parser.add_argument(
            "--ctrl_points_per_interpolation",
            type=int,
            default=1000,
            help="Min step size for knot solver.",
        )
        parser.add_argument(
            "--evaluate_batch_size",
            type=int,
            default=1024,
            help="Min step size for knot solver.",
        )
        parser.add_argument(
            "--random_init_weights",
            action="store_true",
            default=False,
            help="Instead of loading a pretrained model, use a randomly initialized model.",
        )
        parser.add_argument(
            "--plot_knots",
            action="store_true",
            default=False,
            help="Plot polygons at end",
        )
    else:
        # trainer
        parser.add_argument(
            "--save_model",
            action="store_true",
            default=False,
            help="Whether to save model weights to disk.",
        )

    args = parser.parse_args()
    return args
