import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train import RESULTS_PATH


def visualise_10_images(cfg, ds):
    """Show ten images in the dataset

    Args:
        ds (tf dataset)
    """
    image_batch, label_batch = next(iter(ds))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy())
        label = label_batch[i].numpy()
        if cfg["task"] == "regression":
            plt.title(
                "lat: " + str(round(label[0], 2)) + ", long: " + str(round(label[1], 2))
            )
        else:
            plt.title(cfg["classes"][np.argmax(label)])
        plt.axis("off")


def visualise_error_per_epoch(cfg, train_type):
    history = pd.read_csv(RESULTS_PATH + cfg["path"] + train_type + ".csv")

    plt.plot(
        history["epoch"],
        history["mean_squared_error"],
        label="train",
        c="darkgreen",
    )
    plt.plot(
        history["epoch"],
        history["val_mean_squared_error"],
        label="val",
        c="darkblue",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.xlim([history["epoch"].min() - 0.5, history["epoch"].max() + 0.5])
    plt.show()


# TODO: confusion matrix
