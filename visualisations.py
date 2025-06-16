import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from train import RESULTS_PATH, test_run


def plot_10_images(cfg, ds):
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


def plot_error_per_epoch(cfg, train_type):
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


def plot_confusion_matrix(cfg, train_type, mct, tflite):
    y_true, y_pred = test_run(cfg, train_type, mct, tflite)
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(
        y_true_labels, y_pred_labels, labels=[i for i in range(len(cfg["classes"]))]
    )
    plt.figure()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.imshow(cm, interpolation="nearest", cmap="Greens")
    plt.colorbar()
    plt.show()
