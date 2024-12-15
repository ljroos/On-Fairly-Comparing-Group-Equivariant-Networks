import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torchinfo import summary

sys.path.append("./")  # script needs to be run from main directory.

import utilities
from classifier import ComposedCPWAClassifier

from data.image_datasets import CIFARDataModule, MNISTDataModule
from data.image_datasets import plot_batch as batch_image_plot
from data.toy_datasets import MoonsDataModule
from data.toy_datasets import plot_batch as toy_plot

fig_path = utilities.get_fig_path()
save_path = utilities.get_save_path()


def main(args: argparse.Namespace):
    # seed
    if args.aa_seed < 0:
        args.aa_seed = np.random.randint(int(1e7))
    pl.seed_everything(args.aa_seed)

    # logger
    if args.log_results:
        wandb_logger = pl.loggers.WandbLogger(
            entity="ljroos-msc",
            project="knot-solver",
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
    # callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        verbose=True,
        patience=args.patience,
    )

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
            num_train=min(args.num_train, 50000),
            pad_for_rotation=False,
            num_val=0,
            num_test=10000,
            seed=args.aa_seed,
        )
    else:
        raise NotImplementedError

    dm.setup(stage=None)
    dummy_x, dummy_y = next(iter(dm.train_dataloader()))

    model.eval()
    print(
        summary(
            model,
            model.augment(dummy_x).shape,
            device="cuda" if not args.cpu else "cpu",
        )
    )

    if args.dataset == "moons":
        _, ax = plt.subplots(1, 1)

        for i in range(10):
            x, y = next(iter(dm.test_dataloader()))
            toy_plot(
                x,
                y,
                ax=ax,
                show_plot=False,
                save_fig=False,
                remove_lines=True,
            )

        ctrl_points = torch.tensor(
            # [[0, 1.5], [1, 0.5], [-1, 1], [2, 1], [0, 1.5]],  old ctrl points; zig-ish
            [[0, 1.5], [-1, 1], [1, 0.5], [2, 1], [0, 1.5]],
            dtype=torch.float32,
        )
        xs = ctrl_points[:, 0]
        ys = ctrl_points[:, 1]

        # plot arrows out of each point using delta_x
        for x, y in zip(xs, ys):
            plt.plot(
                xs.detach().cpu(),
                ys.detach().cpu(),
                c="gray",
                linestyle="-.",
                linewidth=0.5,
            )

        plt.savefig(
            os.path.join(fig_path, "moons_sample.ignore.png"),
            dpi=450,
            bbox_inches="tight",
        )
        plt.close()

    elif (
        args.dataset == "mnist"
        or args.dataset == "downsampled_mnist"
        or args.dataset == "cifar10"
    ):
        batch_image_plot(
            ims=model.augment(dummy_x),
            labels=dummy_y,
            save_fig=True,
            fig_name=os.path.join(
                fig_path,
                f"{args.dataset}_dataset.ignore.png",
            ),
        )
    else:
        raise NotImplementedError

    # trainer
    trainer = pl.Trainer(
        accelerator="cuda" if not args.cpu else "cpu",
        max_epochs=args.max_epochs,
        callbacks=early_stopping,
        logger=wandb_logger,
        gradient_clip_val=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )

    # fit and test
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.train_dataloader(),
    )
    trainer.validate(model, dm.train_dataloader())
    trainer.test(model, dm)

    # save model
    utilities.save_model(model, args, save_path)

    # sanity check equivariance
    if args.sanity_check_equivariance:
        model.sanity_check_equivariance(dummy_x)


if __name__ == "__main__":
    args = utilities.parse_args()
    main(args)
