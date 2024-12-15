import sys
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy

sys.path.append("./")

from models.classifiers.augmentations import (
    DownsampleWrappedTranslateAugment,
    O2ImageAugment,
)
from models.classifiers.cohen_cnn import P4MCNN
from models.classifiers.downsampled_mnist import DownsampledMNISTClassifier
from models.classifiers.moons import MoonsClassifier
from models.layers.equivariant.group_definitions import D4_action_on_f_z2


class ComposedCPWAClassifier(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
        augment: str,
        model_hparams: dict,
        optimizer_hparams: dict,
    ) -> None:
        super().__init__()
        self.dataset = dataset

        # build classifier
        if dataset == "moons":
            assert augment in ["trivial", "flipH", "flipH_and_or_flipW"]

            self.model = MoonsClassifier(
                group=model_hparams["group"],
                hidden_group_channels=model_hparams["hidden_group_channels"],
                hidden_layers=model_hparams["hidden_layers"],
                batch_norm=model_hparams["batch_norm"],
                dropout=model_hparams["dropout"],
            )

            # augment is already included in dataset
            self.augment = nn.Identity()

        elif dataset in ["mnist", "cifar10"]:
            if dataset == "mnist":
                assert augment in ["trivial", "flipH", "rot", "rotflip"]
                self.augment = O2ImageAugment(augment=augment, padding_mode="border")
            else:
                assert augment in [
                    "trivial",
                    "flipH",
                    "flipW",
                    "rot90",
                    "rot90flip",
                ], "Only allow discrete rotations for CIFAR10."

            self.augment = O2ImageAugment(augment=augment, padding_mode="border")

            self.model = P4MCNN(
                group=model_hparams["group"],
                hidden_group_channels=model_hparams["hidden_group_channels"],
                hidden_layers=model_hparams["hidden_layers"],
                dropout_p=model_hparams["dropout"],
                batch_norm=model_hparams["batch_norm"],
                in_channels=1 if dataset == "mnist" else 3,
            )
        elif dataset == "downsampled_mnist":
            assert augment in [
                "trivial",
                "translateH",
                "translateH_and_or_W",
            ]
            _mode_dict = {
                "trivial": "trivial",
                "translateH": "H",
                "translateW": "W",
                "translateH_and_W": "D",
                "translateH_and_or_W": "HW",
            }
            self.augment = DownsampleWrappedTranslateAugment(mode=_mode_dict[augment])

            self.model = DownsampledMNISTClassifier(
                group=model_hparams["group"],
                hidden_group_channels=model_hparams["hidden_group_channels"],
                hidden_layers=model_hparams["hidden_layers"],
                batch_norm=model_hparams["batch_norm"],
                dropout=model_hparams["dropout"],
            )
        else:
            raise NotImplementedError

        if dataset == "moons":
            self.accuracy = Accuracy(task="binary")
            self.averaged_accuracy = Accuracy(task="binary")
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=10)
            self.averaged_accuracy = Accuracy(task="multiclass", num_classes=10)
            self.loss = nn.CrossEntropyLoss()

        self.optimizer_hparams = optimizer_hparams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_hparams["lr"],
            betas=self.optimizer_hparams["betas"],
            weight_decay=self.optimizer_hparams["weight_decay"],
        )

    def step(self, batch: Any, mode: str):
        x, y = batch

        # for loss calculation
        if self.dataset == "moons":
            y = y.float()

        # augment batch
        x = self.augment(x)

        # forward pass
        y_hat = self(x)
        loss = self.loss(input=y_hat, target=y)

        self.log(f"{mode}_loss", loss)
        self.accuracy(y_hat, y)
        self.log(f"{mode}_accuracy", self.accuracy)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, "val")
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, "test")
        return loss

    def sanity_check_equivariance(self, x: torch.Tensor) -> None:
        if self.dataset == "moons":
            self.sanity_check_equivariance_moons(x)
        elif self.dataset == "downsampled_mnist":
            self.sanity_check_equivariance_downsampled_mnist(x)
        elif self.dataset == "mnist":
            self.sanity_check_equivariance_mnist(x)
        elif self.dataset == "cifar10":
            self.sanity_check_equivariance_mnist(x)

    def sanity_check_equivariance_moons(self, x: torch.Tensor) -> None:
        # remember training state
        train_state = self.training
        self.eval()

        x = self.augment(x)
        with torch.no_grad():
            # pad every label to size 5
            labels = ["trivial", "flipH", "flipW", "rot180"]
            actions = {
                "trivial": lambda x: x,
                "flipH": lambda x: x * torch.tensor([[1, -1]], device=x.device),
                "flipW": lambda x: x * torch.tensor([[-1, 1]], device=x.device),
                "rot180": lambda x: x * torch.tensor([[-1, -1]], device=x.device),
            }
            print_labels = [label + " " * (5 - len(label)) for label in labels]
            group_ims = [actions[label](x) for label in labels]
            print("-" * 24)
            print("Equivariance sanity check:")
            print("percentage neurons within error:")
            print("-" * 24)
            print(
                "operation \t\t 1e-12 \t 1e-10 \t 1e-8 \t 1e-6 \t 1e-5 \t 1e-4 \t 1e-3 \t 1e-2 \t 1e-1"
            )
            for i in range(len(group_ims)):
                print(f"{print_labels[i]}:   ", end="\t")
                transformed_ims = group_ims[i].contiguous()
                out = self.model(x)
                transformed_out = self.model(transformed_ims)
                for eps in [10**n for n in [-12, -10, -8, -6, -5, -4, -3, -2, -1]]:
                    print(
                        f"\t{((out  - transformed_out)**2 < eps).float().mean().item():.4f}",
                        end="",
                    )
                print()
            print("-" * 24)
        self.train(train_state)

    def sanity_check_equivariance_mnist(self, x: torch.Tensor) -> None:
        # remember training state
        train_state = self.training
        self.eval()

        x = self.augment(x)
        with torch.no_grad():
            # pad every label to size 5
            labels = list(D4_action_on_f_z2.keys())
            print_labels = [label + " " * (5 - len(label)) for label in labels]
            group_ims = [D4_action_on_f_z2[label](x) for label in labels]
            print("-" * 24)
            print("Equivariance sanity check:")
            print("percentage neurons within error:")
            print("-" * 24)
            print(
                "operation \t\t 1e-12 \t 1e-10 \t 1e-8 \t 1e-6 \t 1e-5 \t 1e-4 \t 1e-3 \t 1e-2 \t 1e-1"
            )
            for i in range(len(group_ims)):
                print(f"{print_labels[i]}:   ", end="\t")
                transformed_ims = group_ims[i].contiguous()
                out = self.model(x)
                transformed_out = self.model(transformed_ims)
                for eps in [10**n for n in [-12, -10, -8, -6, -5, -4, -3, -2, -1]]:
                    print(
                        f"\t{((out  - transformed_out)**2 < eps).float().mean().item():.4f}",
                        end="",
                    )
                print()
            print("-" * 24)
        self.train(train_state)

    def sanity_check_equivariance_downsampled_mnist(self, x: torch.Tensor) -> None:
        # remember training state
        train_state = self.training
        self.eval()

        with torch.no_grad():
            # pad every label to size 5
            labels = ["trivial", "H", "W", "HW", "D"]
            print_labels = [label + " " * (5 - len(label)) for label in labels]
            print("-" * 24)
            print("Equivariance sanity check:")
            print("percentage neurons within error:")
            print("-" * 24)
            print(
                "random operation \t 1e-12 \t 1e-10 \t 1e-8 \t 1e-6 \t 1e-5 \t 1e-4 \t 1e-3 \t 1e-2 \t 1e-1"
            )
            original_out = self(self.augment.downsample(x))
            for i in range(len(labels)):
                print(f"{print_labels[i]}:   ", end="\t")
                augmenter = DownsampleWrappedTranslateAugment(mode=labels[i])
                translated_out = self(augmenter(x))
                for eps in [10**n for n in [-12, -10, -8, -6, -5, -4, -3, -2, -1]]:
                    print(
                        f"\t{((translated_out  - original_out)**2 < eps).float().mean().item():.4f}",
                        end="",
                    )
                print()
            print("-" * 24)
        self.train(train_state)
