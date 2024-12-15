import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, Pad, ToTensor


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 256,
        num_train: int = 50000,
        num_val: int = None,
        num_test: int = None,
        variant: str = "original",
        pad_for_rotation: bool = False,
        seed: int = 42,
        random_labels: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed
        self.random_labels = random_labels

        transforms = [
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,)),
        ]  # images scaled to [-1, 1]
        if pad_for_rotation:
            transforms.append(Pad(6, -1))

        self.transform = Compose(transforms)

        datamodules = {
            "original": datasets.MNIST,
            "fashion": datasets.FashionMNIST,
            "extended": datasets.EMNIST,
            "kuzushiji": datasets.KMNIST,
            "qmnist": datasets.QMNIST,
        }
        self.Dataset = datamodules[variant]

    def prepare_data(self):
        # download
        self.Dataset(self.data_dir, train=True, download=True)
        self.Dataset(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # don't need to use stages since MNIST is small enough to fit into memory.
        train = self.Dataset(
            self.data_dir, train=True, transform=self.transform, download=True
        )
        test = self.Dataset(
            self.data_dir, train=False, transform=self.transform, download=True
        )
        generator = torch.Generator().manual_seed(self.seed)
        if self.random_labels:
            train.targets = torch.randint(
                0, 10, (len(train.targets),), generator=generator
            )
            test.targets = torch.randint(
                0, 10, (len(test.targets),), generator=generator
            )
        full = torch.utils.data.ConcatDataset([train, test])

        if self.num_val is None:
            self.num_val = self.num_train // 5
        if self.num_test is None:
            self.num_test = len(full) - self.num_train - self.num_val

        num_discard = len(full) - (self.num_train + self.num_val + self.num_test)
        self.train, self.val, self.test, _ = random_split(
            dataset=full,
            lengths=[
                self.num_train,
                self.num_val,
                self.num_test,
                num_discard,
            ],
            generator=generator,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train if self.random_labels else self.val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.train if self.random_labels else self.test, batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train if self.random_labels else self.test, batch_size=self.batch_size
        )


class CIFARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 256,
        num_train: int = 45000,
        num_val: int = None,
        num_test: int = None,
        pad_for_rotation: bool = True,
        variant: str = "10",
        seed: int = 42,
        random_labels: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed
        self.num_train = num_train
        self.num_val = num_val
        self.random_labels = random_labels

        cifar_mean = (0.491, 0.482, 0.447)
        cifar_std = (0.247, 0.243, 0.262)
        cifar_mean = (0.5, 0.5, 0.5)
        cifar_std = (0.5, 0.5, 0.5)
        transformations = [ToTensor(), Normalize(cifar_mean, cifar_std)]
        if pad_for_rotation:  # ensures that the image is not cropped after rotation
            transformations.append(Pad(7))
        self.transform = Compose(transformations)

        assert variant in ["10", "100"], "version must be '10' or '100'"
        self.Dataset = datasets.CIFAR10 if variant == "10" else datasets.CIFAR100

    def prepare_data(self):
        # download
        self.Dataset(self.data_dir, train=True, download=True)
        self.Dataset(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # don't need to use stages since MNIST is small enough to fit into memory.
        train = self.Dataset(
            self.data_dir, train=True, transform=self.transform, download=True
        )
        test = self.Dataset(
            self.data_dir, train=False, transform=self.transform, download=True
        )
        generator = torch.Generator().manual_seed(self.seed)
        if self.random_labels:
            train.targets = torch.randint(
                0, 10, (len(train.targets),), generator=generator
            )
            test.targets = torch.randint(
                0, 10, (len(test.targets),), generator=generator
            )
        full = torch.utils.data.ConcatDataset([train, test])

        if self.num_val is None:
            self.num_val = self.num_train // 5
        if self.num_test is None:
            self.num_test = len(full) - self.num_train - self.num_val

        generator = torch.Generator().manual_seed(self.seed)
        num_discard = len(full) - (self.num_train + self.num_val + self.num_test)
        self.train, self.val, self.test, _ = random_split(
            dataset=full,
            lengths=[
                self.num_train,
                self.num_val,
                self.num_test,
                num_discard,
            ],
            generator=generator,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train if self.random_labels else self.val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.train if self.random_labels else self.test, batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return DataLoader(
            self.train if self.random_labels else self.test, batch_size=self.batch_size
        )


def plot_batch(ims, labels=None, grid_side_len=4, save_fig=False, fig_name=None):
    ims = ims.detach().clamp(-1, 1).cpu()
    if labels is not None:
        labels = labels.detach().cpu()

    # if grid_side_len is a tuple:
    if isinstance(grid_side_len, tuple):
        grid_height, grid_width = grid_side_len
    else:
        grid_width = grid_side_len
        grid_height = grid_side_len
    fig, ax = plt.subplots(grid_height, grid_width, figsize=(10, 10))

    for i in range(grid_height):
        for j in range(grid_width):
            k = grid_width * i + j
            ax[i, j].imshow(
                ims[k].permute(1, 2, 0).clamp(-1, 1) / 2 + 0.5, cmap="inferno"
            )
            if labels is not None:
                ax[i, j].set_title(f"{labels[k]}")
            ax[i, j].axis("off")
    plt.tight_layout()
    if save_fig:
        assert fig_name is not None, "must provide fig_name"
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()
