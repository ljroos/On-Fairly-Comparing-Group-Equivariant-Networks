import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

_tour_tuple = (0, 8, 2, 5, 3, 7, 6, 4, 1, 9, 0)  # see explore vae.ipynb


def _rsample_normal(mu, logvar, sample_max_prob=False):
    if sample_max_prob:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _negative_normal_log_likelihood(x, mu, logvar, reduce="sum"):
    # x, mu, logvar, shape all the same.
    # shape: (batch_size, *dims)
    l = 0.5 * (
        torch.log(torch.tensor(2 * torch.pi)) + logvar + (x - mu) ** 2 / logvar.exp()
    )
    if reduce == "sum":
        return l.sum()
    elif reduce == "none":
        return l


# look up formula KL divergence between diagonal and std normal wikipedia
def _kl_normal_bar_std_normal(mu, logvar):
    # mu, logvar shape the same.
    # KL divergence between N(mu, logvar) and N(0, I)
    return (logvar.exp() + mu**2 - 1 - logvar).sum() / 2


class PrintDebug(nn.Module):
    def __init__(self, breakpoint: bool = False) -> None:
        super().__init__()
        self.breakpoint = breakpoint

    def forward(self, x):
        # if x is a tuple, print the shape of each element
        if isinstance(x, tuple):
            for i, elem in enumerate(x):
                print(f"elem {i}: {elem.shape}")
                # print whether elem contains nan
                print(torch.isnan(elem).any())
        else:
            print(x.shape)
            # print whether x contains nan
            print(torch.isnan(x).any())
        return x


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(-1, *self.target_shape)


class Chunk(nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, x):
        return x.chunk(self.chunk_size, dim=1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        relu=True,
        batch_norm=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.relu = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VariationalAutoencoder(pl.LightningModule):
    def __init__(self, kl_factor=1.0, num_labels=10):
        super().__init__()
        self.kl_factor = kl_factor
        self.num_labels = num_labels

        in_channels = 3
        latent_channels = 32
        hidden_channels = 128

        self.latent_shape = (latent_channels, 4, 4)

        # encoder: function taking x, returning mu_z, logvar_z
        # decoder: function taking z, returning mu_x, logvar_x
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, 3, 1, 1),
            ConvBlock(hidden_channels, hidden_channels, 4, 2, 1),
            ConvBlock(hidden_channels, hidden_channels, 4, 2, 1),
            ConvBlock(
                hidden_channels,
                2 * latent_channels,
                4,
                2,
                1,
                relu=False,
                batch_norm=False,
            ),
            Chunk(2),
        )

        self.decoder = nn.Sequential(
            ConvBlock(latent_channels, hidden_channels, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(hidden_channels, hidden_channels, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(hidden_channels, hidden_channels, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(
                hidden_channels, 2 * in_channels, 3, 1, 1, relu=False, batch_norm=False
            ),
            Chunk(2),
        )

        self.mean_mu_encodings = nn.Parameter(
            torch.zeros((self.num_labels, *self.latent_shape)),
            requires_grad=False,
        )  # mean encodings of each class
        self.mean_logvar_encodings = nn.Parameter(
            torch.zeros((self.num_labels, *self.latent_shape)),
            requires_grad=False,
        )  # mean encodings of each class
        self.best_encodings = nn.Parameter(
            torch.zeros((self.num_labels, *self.latent_shape)),
            requires_grad=False,
        )  # mean encodings of each class

        self.save_hyperparameters()

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = _rsample_normal(mu_z, logvar_z)
        mu_x, logvar_x = self.decoder(z)
        return mu_x, logvar_x, mu_z, logvar_z

    def sample_z(self, batch_size):
        return torch.randn((batch_size, *self.latent_shape))

    def sample_x(self, batch_size):
        z = self.sample_z(batch_size)
        mu_x, lgovar_x = self.decoder(z)
        x = _rsample_normal(mu_x, lgovar_x)
        return x

    def reconstruct(self, x):
        # reconstructs x using maximum a posteriori estimate for both z and x
        mu_z, _ = self.encoder(x)
        mu_x, _ = self.decoder(mu_z)
        return mu_x

    def training_step(self, batch, batch_idx):
        return self.step("train", batch, batch_idx, augment=True)

    def validation_step(self, batch, batch_idx):
        return self.step("val", batch, batch_idx, augment=False)

    def test_step(self, batch, batch_idx):
        return self.step("test", batch, batch_idx, augment=False)

    def step(self, name: str, batch, batch_idx, augment=True):
        x, _ = batch

        # MSE reconstruction loss
        mu_x, logvar_x, mu_z, logvar_z = self(x)

        # standardize both kl and reconstr by im dim
        dim = np.prod(x.shape)
        kl_term = _kl_normal_bar_std_normal(mu_z, logvar_z) / dim
        reconstr_term = _negative_normal_log_likelihood(x, mu_x, logvar_x) / dim
        loss = reconstr_term + self.kl_factor * kl_term
        elbo = -(reconstr_term + kl_term)

        self.log(f"{name}_reconstr_term", reconstr_term)
        self.log(f"{name}_kl_term", kl_term)
        self.log(f"{name}_loss", loss)
        self.log(f"{name}_elbo", elbo)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def compute_best_encodings(self, dataloader):
        """
        Compute the mean embedding for each class in the given dataset.
        Saves the mean embeddings as a tensor in self.

        Not 100% exact--doesn't necessarily weight samples exactly--but good enough for our purposes.
        """
        self.mean_mu_encodings.zero_()
        self.mean_logvar_encodings.zero_()
        with torch.no_grad():
            y_counts = torch.zeros(self.num_labels)
            for x, y in dataloader:
                y_counts += y.bincount(minlength=self.num_labels)

                mu_z, logvar_z = self.encoder(x.to(self.device))

                # calculate the mean encoding of every class
                for i in range(self.num_labels):
                    if (y == i).sum() == 0:
                        continue
                    self.mean_mu_encodings[i] += mu_z[y == i].sum(dim=0)
                    self.mean_logvar_encodings[i] += logvar_z[y == i].sum(dim=0)

            # normalize the mean encodings
            self.mean_mu_encodings /= y_counts.view(
                self.num_labels, *([1] * len(self.mean_mu_encodings.shape[1:]))
            ).to(self.device)
            self.mean_logvar_encodings /= y_counts.view(
                self.num_labels, *([1] * len(self.mean_logvar_encodings.shape[1:]))
            ).to(self.device)

            lowest_recon_loss = torch.full((self.num_labels,), float("inf"))
            for x, y in dataloader:
                mu_z, logvar_z = self.encoder(x.to(self.device))

                for i in range(self.num_labels):
                    # calculate the reconstruction loss of every class
                    class_idx = y == i
                    if class_idx.sum() == 0:
                        continue
                    recon_loss = (
                        _negative_normal_log_likelihood(
                            x=mu_z[class_idx],
                            mu=self.mean_mu_encodings[i],
                            logvar=self.mean_logvar_encodings[i],
                            reduce="none",
                        )
                        .flatten(1)
                        .mean(1)
                    )
                    lowest_recon_value, lowest_recon_idx = torch.min(recon_loss, dim=0)

                    if lowest_recon_value < lowest_recon_loss[i]:
                        lowest_recon_loss[i] = lowest_recon_value
                        self.best_encodings[i] = mu_z[class_idx][lowest_recon_idx]

    def interpolate_from_encodings(self, z_start, z_end, num_interps, mode="linear"):
        """
        Interpolate between two encodings.
        z_start and z_end both have batch_dim = 1
        """
        with torch.no_grad():
            device, dtype = z_start.device, z_start.dtype
            num_dims = len(z_start.shape)
            if mode == "linear":
                t = torch.linspace(0, 1, num_interps, dtype=dtype, device=device)

                # reshape t for broadcasting
                t = t.view(-1, *([1] * (num_dims)))

                z_interps = z_start * (1 - t) + z_end * t
            else:
                raise NotImplementedError

            x_interps, _ = self.decoder(z_interps)

        return z_interps, x_interps
