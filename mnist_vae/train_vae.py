import datetime
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append("./")  # script needs to be run from main directory.

from vae import VariationalAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pl.seed_everything(42)


# Load MNIST dataset
transform = transforms.ToTensor()
mnist_train_val = datasets.MNIST(
    root="raw-data/", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="raw-data/", train=False, download=True, transform=transform
)
# Split MNIST data into train, validation, and test sets
train_data, val_data = random_split(mnist_train_val, [50000, 10000])

BATCH_SIZE = 2048
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Initialize the autoencoder and trainer
model = VariationalAutoencoder()

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=5,
    verbose=True,
    mode="min",
)
trainer = pl.Trainer(
    max_epochs=-1,
    callbacks=[early_stop_callback],
    deterministic=True,
    log_every_n_steps=25,
)

# Train the autoencoder
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

# Have the model compute its mean embeddings for every digit on the test set
model.eval()
model.to(device)
model.compute_mean_embeddings(test_loader)


# save the model
save_path = os.path.join("mnist_vae", "trained_vae_state_dict.pt")

if os.path.exists(save_path):
    print("Model already exists. Adding a timestamp to the filename.")
    save_path = os.path.join(
        "mnist_vae",
        f"trained_vae_state_dict_{datetime.datetime.now()}.pt",
    )

torch.save(model.state_dict(), save_path)
