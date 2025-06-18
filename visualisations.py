import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from train import RESULTS_PATH, test_run

PLOT_FOLDER = "/home/kaptim/eth/mlmc/project/bottom_up/code/plots/"


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


def plot_test_accuracy_sorted():
    tests = pd.read_csv(RESULTS_PATH + "test_results.csv")
    # include qat information
    # insert dummy 0 accuracy for non-quantized qat models
    tests_dict = tests.to_dict(orient="records")
    additional_rows = []
    for test in tests_dict:
        if test["QAT"] == "qat" and test["TFLITE"] == True:
            additional_rows.append(
                {
                    k: (0 if k == "Accuracy" else (False if k == "TFLITE" else v))
                    for k, v in test.items()
                }
            )
    tests = pd.DataFrame(tests_dict + additional_rows)
    tests["Name"] = tests["Name"] + tests["QAT"].fillna("")

    tests_q = (
        tests[tests["TFLITE"] == True]
        .sort_values(by="Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    tests_non_q = tests[tests["TFLITE"] == False]
    # sort non-quantised values based on quantised values
    tests_non_q["q_ranking"] = [
        tests_q[tests_q["Name"] == x].index[0] for x in tests_non_q["Name"].to_list()
    ]
    tests_non_q = tests_non_q.sort_values(by="q_ranking").drop("q_ranking", axis=1)

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    x = [
        "".join([c for c in l.split("_")[0] if c.isupper() or c.isnumeric()])
        + "\n"
        + "_".join(l.split("_")[1:])
        for l in tests_q.Name.to_list()
    ]
    y1 = tests_non_q.Accuracy.to_numpy()
    ax1.set_ylabel("Floating Point")
    ax1.bar(x, y1, color="darkblue")
    y2 = tests_q.Accuracy.to_numpy()
    ax2.set_ylabel("Quantised")
    ax2.bar(x, y2, color="darkgreen")
    ax2.tick_params(axis="x", labelsize=6)
    f.suptitle("Test Accuracy", fontsize=14)

    f.savefig(PLOT_FOLDER + "test_accuracy.png")
