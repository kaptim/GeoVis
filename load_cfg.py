import yaml
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras as keras
from load_data import fit_oh_encoder

CFGS_FOLDER = "/home/kaptim/eth/mlmc/project/bottom_up/code/cfgs"


def convert_loss(loss_str):
    if loss_str == "mse":
        return keras.losses.MSE
    else:
        raise ValueError(loss_str + " not a known loss")


def convert_metric(metric_str):
    if metric_str == "mse":
        return keras.metrics.MeanSquaredError()
    elif metric_str == "mae":
        return keras.metrics.MeanAbsoluteError()
    else:
        raise ValueError(metric_str + " not a known metric")


def convert_optimizer(optimizer_str, lr):
    if optimizer_str == "adam":
        return keras.optimizers.Adam(lr)
    else:
        raise ValueError(optimizer_str + " not a known optimizer")


def convert_preprocessor(preprocessor_str):
    if preprocessor_str == "mobile_net_v2":
        return keras.applications.mobilenet_v2.preprocess_input
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


def load_cfg(cfg_path):
    # cfg_path supplied when running main
    # initialise checkpoint path, set up classes (e.g., MSE loss class instead of mse)
    with open(CFGS_FOLDER + "/" + cfg_path) as stream:
        cfg = yaml.safe_load(stream)
    cfg["path"] = cfg_path.split(".")[0]
    cfg["loss"] = convert_loss(cfg["loss"])
    cfg["metrics"] = [convert_metric(metric) for metric in cfg["metrics"]]
    cfg["optimizer"] = convert_optimizer(cfg["optimizer"], cfg["lr"])
    cfg["preprocessor"] = convert_preprocessor(cfg.get("preprocessor", None))
    add_oh_encoder(cfg)
    check_targets(cfg)
    return cfg
