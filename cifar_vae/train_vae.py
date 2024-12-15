import datetime
import os
import sys

import pytorch_lightning as pl
import torch
import torchinfo
from pytorch_lightning.callbacks import EarlyStopping

sys.path.append("./")  # script needs to be run from main directory.

from cifar_vae.vae import VariationalAutoencoder
from data.image_datasets import CIFARDataModule, plot_batch

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
pl.seed_everything(42)

dm = CIFARDataModule(
    data_dir="./raw-data/",
    batch_size=128,
    num_train=45000,
    num_val=5000,
    num_test=10000,
    pad_for_rotation=False,
    variant="10",
    seed=42,
    random_labels=False,
)
dm.setup(None)


# Initialize the autoencoder and trainer
model = VariationalAutoencoder(num_labels=10, kl_factor=1)

x = next(iter(dm.train_dataloader()))[0]
torchinfo.summary(model, input_data=x, device=device)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=5,
    verbose=True,
    mode="min",
)


trainer = pl.Trainer(
    max_epochs=18,
    accelerator=device_name,
    callbacks=[early_stop_callback],
    log_every_n_steps=25,
    gradient_clip_val=1,
)

# Train the autoencoder
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)

# Have the model compute its mean embeddings for every digit on the test set
model.to(device)
model.eval()
model.compute_best_encodings(dm.test_dataloader())
train_batch = next(iter(dm.train_dataloader()))[0]
test_batch = next(iter(dm.test_dataloader()))[0]
with torch.no_grad():
    best_decodings = model.decoder(model.best_encodings)[0].clamp(-1, 1).cpu()
    train_reconstructions = (
        model.reconstruct(train_batch.to(model.device)).clamp(-1, 1).cpu()
    )
    test_reconstructions = (
        model.reconstruct(test_batch.to(model.device)).clamp(-1, 1).cpu()
    )
model.cpu()

# if cifar_vae/figs does not exist, create it
if not os.path.exists(os.path.join("cifar_vae", "figs")):
    os.makedirs(os.path.join("cifar_vae", "figs"))

figs_dir = os.path.join("cifar_vae", "figs")

plot_batch(
    best_decodings,
    grid_side_len=(2, 5),
    save_fig=True,
    fig_name=os.path.join(figs_dir, "best_decodings.png"),
)

plot_batch(
    test_batch,
    grid_side_len=(2, 5),
    save_fig=True,
    fig_name=os.path.join(figs_dir, "test_batch.png"),
)

plot_batch(
    train_batch,
    grid_side_len=(2, 5),
    save_fig=True,
    fig_name=os.path.join(figs_dir, "train_batch.png"),
)

plot_batch(
    train_reconstructions,
    grid_side_len=(2, 5),
    save_fig=True,
    fig_name=os.path.join(figs_dir, "train_reconstructions.png"),
)

plot_batch(
    test_reconstructions,
    grid_side_len=(2, 5),
    save_fig=True,
    fig_name=os.path.join(figs_dir, "test_reconstructions.png"),
)

save_path = os.path.join("cifar_vae", "trained_vae_state_dict.pt")
if os.path.exists(save_path):
    print("Model already exists. Adding a timestamp to the filename.")
    save_path = os.path.join(
        "cifar_vae",
        f"trained_vae_state_dict_{datetime.datetime.now()}.pt",
    )

trainer.save_checkpoint(save_path)
