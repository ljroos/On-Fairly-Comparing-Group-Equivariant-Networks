import os

SAVE_FOLDER = "csv-files"

# CohenCNN rotflipMNIST
MNIST_POLY_SWEEP = "ljroos-msc/mnist-polygons/kk4268dd"
MNIST_POLY_SAVE_LOC = os.path.join(
    SAVE_FOLDER, "mnist_polygons_just_group_channels.csv"
)

# downsampled MNIST
DOWNSAMPLED_MNIST_POLY_SWEEP = "ljroos-msc/downsampled-mnist-polygons/yxsk6yqx"
DOWNSAMPLED_MNIST_POLY_SAVE_LOC = os.path.join(SAVE_FOLDER, "downsampled_mnist_.csv")
# downsampled MNIST WITHOUT batch-norm
NO_BATCH_DOWNSAMPLED_MNIST_POLY_SWEEP = "ljroos-msc/downsampled-mnist-polygons/dqv058zj"
NO_BATCH_DOWNSAMPLED_MNIST_POLY_SAVE_LOC = os.path.join(
    SAVE_FOLDER, "no_batch_downsampled_mnist_.csv"
)

# toy problem
MOONS_MOSAIC_SWEEP = "ljroos-msc/mosaic/tvcv0yl4"
MOONS_MOSAIC_SAVE_LOC = os.path.join(SAVE_FOLDER, "moons.csv")

# toy problem
PINWHEEL_MOSAIC_SWEEP = "ljroos-msc/mosaic/w705aehx"
PINWHEEL_MOSAIC_SAVE_LOC = os.path.join(SAVE_FOLDER, "pinwheel.csv")

if __name__ == "__main__":
    pass
