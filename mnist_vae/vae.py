import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        LATENT_DIM = 32  # quick test found best ELBO with this latent dim
        self.latent_dim = LATENT_DIM
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )
        self.fc_mu = nn.Linear(64, self.latent_dim)
        self.fc_logvar = nn.Linear(64, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        self.mean_z = nn.Parameter(
            torch.zeros((10, self.latent_dim)), requires_grad=False
        )  # mean embeddings of each digit class

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x = self.decode(z)
        return x, mu, logvar

    def reconstruct(self, x):  # max likelihood reconstruction
        mu, logvar = self.encode(x)
        x_hat = self.decode(mu)
        return x_hat

    def decode(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        x = self.decoder(z)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def training_step(self, batch, batch_idx):
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step("test", batch, batch_idx)

    def step(self, name: str, batch, batch_idx):
        img, _ = batch
        recon_img, mu, logvar = self(img)
        B = img.shape[0]
        recon_loss = F.binary_cross_entropy(recon_img, img, reduction="sum") / B
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
        loss = recon_loss + kl_divergence
        self.log(f"{name}_recon_loss", recon_loss)
        self.log(f"{name}_kl_divergence", kl_divergence)
        self.log(f"{name}_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def compute_mean_embeddings(self, dataloader):
        """
        Compute the mean embedding for each digit in the given dataset. Saves the mean embeddings as a tensor in self.
        :param dataloader: a DataLoader instance
        :return: None
        """
        # compute the average embedding of every class in MNIST
        with torch.no_grad():
            z = []
            y = []
            for x, label in dataloader:
                mu, _ = self.encode(x.to(self.device))
                z.append(mu)
                y.append(label)

            z = torch.cat(z, dim=0)
            y = torch.cat(y, dim=0)

            # calculate the mean embeddings of every class
            mean_z = []
            for i in range(10):
                mean_z.append(z[y == i].mean(dim=0))
            self.mean_z[:] = torch.stack(mean_z).cpu()
