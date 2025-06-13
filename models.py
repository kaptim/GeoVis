import os

# needed for quantization-aware training in tensorflow > 2.15
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot
from load_data import BATCH_SIZE

MODELS_PATH = "/home/kaptim/eth/mlmc/project/bottom_up/code/saved_models/"
WEIGHTS_PATH = "/home/kaptim/eth/mlmc/project/bottom_up/code/weights/"

# TODO: rebuild with functions + Sequential
# TODO: increase dense layers


def naive_conv_block(model, block_size, filters, kernel, strides, padding):
    for i in range(block_size):
        model.add(keras.layers.Conv2D(filters, kernel, strides, padding))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))


def naive_reg_net(cfg):
    model = keras.Sequential()
    # specify input dimension for .summary()
    model.add(keras.Input(shape=(cfg["img_height"], cfg["img_width"], 3)))
    filters = cfg["filters"]

    for i in range(cfg["num_blocks"]):
        naive_conv_block(
            model,
            cfg["block_size"],
            filters,
            kernel=(3, 3),
            strides=1,
            padding="same",
        )
        model.add(keras.layers.MaxPooling2D((cfg["maxpool"], cfg["maxpool"])))
        model.add(tf.keras.layers.Dropout(cfg["dropout_conv"]))
        filters *= 2

    model.add(keras.layers.AveragePooling2D((cfg["averpool"], cfg["averpool"])))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(cfg["dense-1"], activation="relu"))
    model.add(tf.keras.layers.Dropout(cfg["dropout_dense"]))
    model.add(keras.layers.Dense(2))
    return model


def create_model(cfg, train_type):
    """Create and compiles a keras.Model based on the elements in cfg"""
    model = globals()[cfg["model"]](cfg)
    if train_type == "qat":
        qa_model = tfmot.quantization.keras.quantize_model(model)
        qa_model.compile(
            optimizer=cfg["optimizer"],
            loss=cfg["loss"],
            metrics=cfg["metrics"],
        )
        return qa_model
    else:
        model.compile(
            optimizer=cfg["optimizer"], loss=cfg["loss"], metrics=cfg["metrics"]
        )
        return model


def load_model(cfg, train_type, mct):
    """Create model and load saved weights into it
    train_type: "" if fp, "qat" if qa training
    mct: "mct" if mct (Sony), "" else"""
    model = create_model(cfg, train_type)
    model.load_weights(WEIGHTS_PATH + cfg["path"] + mct + train_type + ".weights.h5")
    return model
