import yaml
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import keras
from load_data import fit_oh_encoder

if tf.__version__ >= "2.19.0":
    # keras_hub needs tensorflow >= 2.19
    from keras_hub.layers import CLIPImageConverter

CFGS_FOLDER = "/home/kaptim/eth/mlmc/project/bottom_up/code/cfgs"


def convert_loss(cfg):
    preprocessor_str = cfg.get("preprocessor", None)
    if preprocessor_str is None or preprocessor_str != "clip":
        if cfg["loss"] == "mse":
            return tf.keras.losses.MSE
        elif cfg["loss"] == "ce":
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            raise ValueError(cfg["loss"] + " not a known loss")
    else:
        if cfg["loss"] == "mse":
            return keras.losses.MeanSquaredError()
        elif cfg["loss"] == "ce":
            return keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            raise ValueError(cfg["loss"] + " not a known loss")


def convert_metric(cfg, metric):
    preprocessor_str = cfg.get("preprocessor", None)
    if preprocessor_str is None or preprocessor_str != "clip":
        if metric == "mse":
            return tf.keras.metrics.MeanSquaredError()
        elif metric == "mae":
            return tf.keras.metrics.MeanAbsoluteError()
        elif metric == "acc":
            return tf.keras.metrics.CategoricalAccuracy()
        else:
            raise ValueError(metric + " not a known metric")
    else:
        if metric == "mse":
            return keras.metrics.MeanSquaredError()
        elif metric == "mae":
            return keras.metrics.MeanAbsoluteError()
        elif metric == "acc":
            return keras.metrics.CategoricalAccuracy()
        else:
            raise ValueError(metric + " not a known metric")


def convert_optimizer(cfg):
    preprocessor_str = cfg.get("preprocessor", None)
    if preprocessor_str is None or preprocessor_str != "clip":
        if cfg["optimizer"] == "adam":
            return tf.keras.optimizers.Adam(cfg["lr"])
        else:
            raise ValueError(cfg["optimizer"] + " not a known optimizer")
    else:
        if cfg["optimizer"] == "adam":
            return keras.optimizers.Adam(cfg["lr"])
        else:
            raise ValueError(cfg["optimizer"] + " not a known optimizer")


def convert_preprocessor(cfg):
    preprocessor_str = cfg.get("preprocessor", None)
    if preprocessor_str == "mobile_net_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    elif preprocessor_str == "clip":
        return CLIPImageConverter.from_preset(cfg["clip_str"])
    else:
        return None


def add_oh_encoder(cfg):
    # set up one_hot_encoder (needed for classification)
    if cfg["task"] == "classification":
        print("Classification task: fit OneHotEncoder on train set")
        cfg["oh_encoder"] = OneHotEncoder(handle_unknown="error", dtype=np.float32)
        # need to always fit it to the training data so that we have
        # the same encoder for training and testing (this may take a while)
        fit_oh_encoder(cfg)
        cfg["classes"] = cfg["oh_encoder"].categories_[0]
    elif cfg["task"] == "regression":
        return
    else:
        raise ValueError(cfg["task"] + " not a known task")


def check_targets(cfg):
    if cfg["task"] == "classification" and len(cfg["targets"]) > 1:
        raise ValueError("More than one target is not supported for classification")


def check_subtask(cfg):
    subtask = cfg.get("subtask", None)
    if (
        subtask not in ["feature-extraction", "fine-tuning", "from-scratch"]
        and subtask is not None
    ):
        raise ValueError(str(subtask) + " not a known task")
    if subtask == "fine-tuning":
        if cfg["fine_tune_at"] not in ["last", "full"]:
            raise ValueError(
                cfg["fine_tune_at"] + " not a known fine-tuning configuration"
            )
        if not cfg["qat"]:
            cfg["train_type"] = ""
        else:
            cfg["train_type"] = "qat"


def load_cfg(cfg_path):
    # cfg_path supplied when running main
    # initialise checkpoint path, set up classes (e.g., MSE loss class instead of mse)
    with open(CFGS_FOLDER + "/" + cfg_path) as stream:
        cfg = yaml.safe_load(stream)
    cfg["path"] = cfg_path.split(".")[0]
    cfg["loss"] = convert_loss(cfg)
    cfg["metrics"] = [convert_metric(cfg, metric) for metric in cfg["metrics"]]
    cfg["optimizer"] = convert_optimizer(cfg)
    cfg["preprocessor"] = convert_preprocessor(cfg)
    add_oh_encoder(cfg)
    check_targets(cfg)
    check_subtask(cfg)
    print(cfg)
    return cfg
