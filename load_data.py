import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow.keras as keras
from get_data import DATA_DIR

PROCESSED_DATA_DIR = "/home/kaptim/eth/mlmc/project/bottom_up/code/processed_data/"

# set seed for reproducibility
tf.random.set_seed(0)


def set_up_classification(cfg, is_scaled=False):
    # fit one-hot encoder on training dataset
    columns = list(set(["id", "country"] + cfg["targets"]))
    metadata = pd.read_csv(DATA_DIR + "/train.csv").loc[:, columns]
    metadata = metadata[metadata["country"].isin(cfg["countries"])]
    cfg["oh_encoder"].fit(metadata[cfg["targets"]])

    if cfg.get("balanced", False):
        region_counts = metadata.region.value_counts()[
            cfg["oh_encoder"].categories_[0].tolist()
        ]
        """ # save initial bias
        if is_scaled:
            bias = (
                -1
                + (region_counts - region_counts.min())
                * 2
                / (region_counts.max() - region_counts.min())
            ).values
        else:
            bias = region_counts.values

        def bias_init(shape):
            return tf.Variable(bias, dtype=np.float32)

        cfg["bias_init"] = bias_init"""
        # save class weights
        cfg["class_weight"] = {
            i: (1 / region_counts.iloc[i]) * (metadata.shape[0] / 2.0)
            for i in range(region_counts.shape[0])
        }


def get_metadata(cfg, is_train: bool):
    # returns the pd dataframe containing the metadata for training the model
    columns = list(set(["id", "country"] + cfg["targets"]))
    if is_train:
        metadata = pd.read_csv(DATA_DIR + "/train.csv").loc[:, columns]
    else:
        metadata = pd.read_csv(DATA_DIR + "/test.csv").loc[:, columns]
    metadata = metadata[metadata["country"].isin(cfg["countries"])]
    if cfg["task"] == "regression":
        return (
            tf.convert_to_tensor(metadata.loc[:, "id"].to_numpy()),
            metadata["id"].astype(str).to_list(),
            tf.convert_to_tensor(metadata.loc[:, cfg["targets"]].to_numpy()),
        )
    else:
        # classification => need to one-hot encode
        return (
            tf.convert_to_tensor(metadata.loc[:, "id"].to_numpy()),
            metadata["id"].astype(str).to_list(),
            tf.convert_to_tensor(
                cfg["oh_encoder"].transform(metadata[cfg["targets"]]).toarray()
            ),
        )


def get_metadata_index(file_path, ids):
    # get index of image in the metadata dataframe based on the image path
    file_name = tf.strings.split(file_path, os.path.sep)[-1]
    file_id = tf.strings.to_number(tf.strings.split(file_name, ".")[0], tf.int64)
    return tf.argmax(file_id == ids)


def get_targets_per_file(file_path, ids, targets):
    # get targets based on id contained in the file name of the image
    return targets[get_metadata_index(file_path, ids), :]


def decode_img(img_height, img_width, img):
    # taken from the tensorflow documentation
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    return img


def process_path(file_path, ids, targets, cfg):
    # taken from the tensorflow documentation: read and decode image
    # function can be applied using a tensorflow map operation
    target = get_targets_per_file(file_path, ids, targets)
    img = tf.io.read_file(file_path)
    img = decode_img(cfg["img_height"], cfg["img_width"], img)
    return img, target


def get_country_data(dataset, cfg, is_train):
    # decode file, resize and get targets per image
    # load data for specific country only
    ids, selected_files, targets_tf = get_metadata(cfg, is_train)
    all_files_dict = {
        f.split(".")[0].split("/")[-1]: f
        for f in tf.io.gfile.glob(DATA_DIR + "/images/" + dataset + "/*/*.jpg")
    }
    # select file names from country
    selected_files_paths = [all_files_dict[f] for f in selected_files]
    ds = tf.data.Dataset.from_tensor_slices(selected_files_paths)
    # decode image, add targets
    ds = ds.map(
        lambda file_path: process_path(file_path, ids, targets_tf, cfg),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


def save_country_data(ds, cfg, name):
    # save tensorflow dataset as numpy arrays (useful for testing on a device)
    x_numpy = np.empty(
        (
            tf.data.experimental.cardinality(ds).numpy(),
            cfg["img_height"],
            cfg["img_width"],
            3,
        )
    )
    y_numpy = np.empty(
        (
            tf.data.experimental.cardinality(ds).numpy(),
            len(cfg["targets"]) if cfg["task"] == "regression" else len(cfg["classes"]),
        )
    )
    for i, data in enumerate(ds):
        x_numpy[i] = data[0].numpy()
        y_numpy[i] = data[1].numpy()

    # full quantization input and output (needed for inference on the device)
    np.save(PROCESSED_DATA_DIR + "x_" + name + "_q", x_numpy.astype(np.uint8))
    np.save(PROCESSED_DATA_DIR + "y_" + name + "_q", y_numpy.astype(np.uint8))


def preprocess_data(ds, cfg, is_train: bool):
    # preprocessing which is necessary for structured training in python
    # rescale RGB values to [0, 1]
    if cfg["preprocessor"] is None:
        # default scaling
        scaling = keras.layers.Rescaling(scale=1.0 / 255)
        ds = ds.map(lambda x, y: (scaling(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # use custom preprocessor for transfer learning
        ds = ds.map(
            lambda x, y: (cfg["preprocessor"](x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if is_train:
        # only shuffle train set
        # set buffer_size for lower memory consumption
        # (no buffer_size set => all data loaded into memory)
        ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(cfg["batch_size"], num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        # only augment train set
        augmentation = tf.keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),  # left-right flip
                keras.layers.RandomRotation(0.1, fill_mode="nearest"),
                keras.layers.RandomCrop(
                    int(cfg["img_height"] * 0.95), int(cfg["img_width"] * 0.95)
                ),
                # random cropping changes the size of the image => need to resize again
                keras.layers.Resizing(cfg["img_height"], cfg["img_width"]),
            ]
        )
        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def load_dataset(cfg, is_train: bool):
    """Preprocesses images and sets up the train, val or test set

    Args:
        cfg (dict):
        is_train (bool): Whether to get training and validation data (True)
            or testing data (False)

    Returns:
        tuple of (train, val) or (test, test_count:int) tf dataset: not loaded into memory
    """
    dataset = "train" if is_train else "test"
    # this step might take a few minutes for train on CPU (linear CPU operation)
    list_ds = get_country_data(dataset, cfg, is_train)

    image_count = tf.data.experimental.cardinality(list_ds).numpy()
    print(str(image_count) + " images in the " + dataset + " set")
    if is_train:
        # split up into train and validation set
        val_size = int(image_count * 0.2)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)
        return preprocess_data(train_ds, cfg, True), preprocess_data(val_ds, cfg, False)
    else:
        return preprocess_data(list_ds, cfg, False), image_count
