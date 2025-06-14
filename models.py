import os

# needed for quantization-aware training in tensorflow > 2.15
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot
import model_compression_toolkit as mct

MODELS_PATH = "/home/kaptim/eth/mlmc/project/bottom_up/code/saved_models/"


def naive_conv_block(model, block_size, filters, kernel, strides, padding):
    for i in range(block_size):
        model.add(keras.layers.Conv2D(filters, kernel, strides, padding))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))


def naive_net(cfg):
    # adaptable version of a CNN with regression or classification output
    model = keras.Sequential()
    # specify input dimension
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
    if cfg["task"] == "regression":
        model.add(keras.layers.Dense(len(cfg["targets"])))
    else:
        # classification: tensorflow discourages to add softmax activation function
        model.add(keras.layers.Dense(len(cfg["classes"])))
    return model


def mobile_net_v2_fe(cfg):
    # mobile net v2 with feature extraction
    # TODO: for regression and classification pretty similar (probably also for naive net)
    img_shape = (cfg["img_height"], cfg["img_width"], 3)
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,  # only the feature extraction layers
        input_shape=img_shape,
    )
    # freeze weights of the feature extractor
    base_model.trainable = False
    # create model
    inputs = keras.Input(shape=img_shape)
    # training=False important in case of batch normalization layers
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(cfg["dropout_dense"])(x)
    if cfg["task"] == "regression":
        outputs = keras.layers.Dense(len(cfg["targets"]))(x)
    else:
        # classification: tensorflow discourages to add softmax activation function
        # (numerical instabilities during training)
        outputs = keras.layers.Dense(len(cfg["classes"]))(x)
    return keras.Model(inputs, outputs)


def create_model(cfg, train_type):
    """Create and compiles a keras.Model based on the values in cfg"""
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
    if not mct:
        try:
            model = create_model(cfg, train_type)
            model.load_weights(
                MODELS_PATH + "weights/" + cfg["path"] + train_type + ".weights.h5"
            )
        except:
            print("Loading .h5 model")
            model = keras.saving.load_model(
                MODELS_PATH + cfg["path"] + train_type + ".h5"
            )
    else:
        print("Loading mct .keras model")
        model = mct.keras_load_quantized_model(
            MODELS_PATH + cfg["path"] + "mct" + train_type + ".keras"
        )
        model.compile(loss=cfg["loss"], metrics=cfg["metrics"])
    return model


def save_h5_model(cfg, train_type, mct):
    """Load and save (trained) tensorflow model
    (.h5: legacy format but very useful for transferring models between tensorflow versions)
    """
    model = load_model(cfg, train_type, mct)
    model.save(MODELS_PATH + cfg["path"] + mct + train_type + ".h5")
    print(".h5 file saved successfully")
