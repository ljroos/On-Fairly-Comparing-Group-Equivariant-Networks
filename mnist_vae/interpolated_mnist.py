import itertools
import os
from itertools import combinations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from .vae import VariationalAutoencoder

VAE_PATH = os.path.join("mnist_vae", "trained_vae_state_dict.pt")


class InterpolatedMNIST(Dataset):
    def __init__(
        self,
        root: str,
        from_digit: int,
        to_digit: int,
        num_interpolations: int,
        transform=None,
        vae_generator_batch_size=2048,
    ):
        assert from_digit != to_digit, "from_digit and to_digit must be different"

        name = "InterpolatedMNIST"
        digits_dir = f"{from_digit}--{to_digit}"
        interp_dir = f"num_interpolations={num_interpolations}"
        self.save_path = os.path.join(root, name, digits_dir, interp_dir)
        self.from_digit = from_digit
        self.to_digit = to_digit
        self.num_interpolations = num_interpolations
        self.batch_size = vae_generator_batch_size

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            self.generate_samples(
                from_digit,
                to_digit,
                num_interpolations,
                self.save_path,
                self.batch_size,
            )

        try:
            self.dataset = torch.empty((num_interpolations, 1, 28, 28))
        except MemoryError:
            raise MemoryError(
                f"Memory error: dataset is too large to fit into memory. Try reducing num_interpolations."
            )
        if len(os.listdir(self.save_path)) == 0:
            raise RuntimeError(
                f"Dataset corrupted error: no samples were generated. Delete the dataset and try again."
            )
        for n in trange(
            len(os.listdir(self.save_path)),
            desc=f"Loading {(from_digit, to_digit)} dataset into memory",
        ):
            begin = n * self.batch_size
            end = min((n + 1) * self.batch_size, num_interpolations)
            self.dataset[begin:end] = torch.load(
                os.path.join(self.save_path, f"batch_{n}.pt"),
                map_location="cpu",
            )
        if end != num_interpolations:
            raise RuntimeError(
                f"Dataset corrupted error: only {end} out of {num_interpolations} samples were loaded. Delete the dataset and try again."
            )

        self.transform = transform
        self.root = root

    def __len__(self):
        return self.num_interpolations

    def __getitem__(self, idx):
        x = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x

    @staticmethod
    def generate_samples(
        from_digit: int,
        to_digit: int,
        num_interpolations: int,
        save_path: str,
        batch_size: int = 1024,
    ):
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load trained VAE
        model = VariationalAutoencoder()
        model.load_state_dict(torch.load(VAE_PATH, map_location=device))
        model.to(device)
        mean_z = model.mean_z  # mean latent embeddings for every class
        with torch.no_grad():
            a = mean_z[from_digit]
            b = mean_z[to_digit]
            timesteps = torch.linspace(0, 1, num_interpolations, device=device)
            # create dataloader out of timesteps
            time_loader = DataLoader(
                timesteps, batch_size=min(batch_size, num_interpolations), shuffle=False
            )

            for n, t in enumerate(
                tqdm(time_loader, desc=f"Generating {(from_digit, to_digit)} samples")
            ):
                z_interp = a * (1 - t[:, None]) + b * t[:, None]
                x_interp = model.decode(z_interp)
                torch.save(x_interp, os.path.join(save_path, f"batch_{n}.pt"))

    @staticmethod
    def get_optimal_digit_pairings(mode="pair"):
        assert mode in ["pair", "min_spanning_tree", "travelling_salesman"]

        # load trained VAE
        model = VariationalAutoencoder()
        model.load_state_dict(torch.load(VAE_PATH))
        mean_z = model.mean_z.numpy()  # mean latent embeddings for every class

        # pairwise distances between embeddings
        dists = np.sqrt(((mean_z[:, None] - mean_z[None, :]) ** 2).sum(axis=-1))

        if mode != "pair":
            raise NotImplementedError("Only the `pair` mode is currently implemented.")

        def generate_pairings(n):
            # Generate a list of 2n elements
            elements = list(range(2 * n))

            # Generate all possible pairings
            pairings = list(combinations(combinations(elements, 2), n))

            # Filter out pairings where elements are repeated
            valid_pairings = [
                pairing
                for pairing in pairings
                if len(set(itertools.chain.from_iterable(pairing))) == 2 * n
            ]

            return valid_pairings

        n = 5  # 2n = 10 elements
        all_pairings = generate_pairings(n)

        # Find the pairing that minimizes the sum of distances
        min_sum = float("inf")
        min_pairing = None

        pair_dists = []
        for pairing in all_pairings:
            # Calculate sum of distances
            sum_dist = sum(dists[i, j] for i, j in pairing)
            pair_dists.append(sum_dist)

            # If this sum is smaller, update min_sum and min_pairing
            if sum_dist < min_sum:
                min_sum = sum_dist
                min_pairing = pairing

        print(f"Minimum sum of distances: {min_sum:.2f}")
        print(f"Minimum sum pairing: {min_pairing}")

        dist_5_percentile = np.percentile(pair_dists, 2.5)
        dist_95_percentile = np.percentile(pair_dists, 97.5)

        # confidence interval for tour dist
        print(
            f"95% distance confidence interval [{dist_5_percentile:.3f},{dist_95_percentile:.3f}]"
        )

        # print the distances between individual pairs
        for pairing in min_pairing:
            print(f"{pairing} -> {dists[pairing[0], pairing[1]]:.2f}")

        return min_pairing


if __name__ == "__main__":
    # Test the class
    print(InterpolatedMNIST.get_optimal_digit_pairings())

    dataset = InterpolatedMNIST(
        root="raw-data/",
        from_digit=0,
        to_digit=1,
        num_interpolations=int(5e6),
        transform=None,
    )

    loader = DataLoader(dataset, batch_size=int(1e5), shuffle=False)

    import matplotlib.pyplot as plt

    # plot the first 10 images
    fig, ax = plt.subplots(1, 10, figsize=(10, 1))
    for i, x in enumerate(tqdm(loader, desc="Plotting images")):
        ax[i].imshow(x[0, 0], cmap="gray")
        ax[i].axis("off")
    plt.show()
