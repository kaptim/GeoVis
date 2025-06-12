import yaml
import tensorflow.keras as keras

CFGS_FOLDER = "/home/kaptim/eth/mlmc/project/bottom_up/code/cfgs"


def convert_loss(loss_str):
    if loss_str == "mse":
        return keras.losses.MSE
    else:
        return ValueError(loss_str + " not a known loss")


def convert_metric(metric_str):
    if metric_str == "mse":
        return keras.metrics.MeanSquaredError()
    elif metric_str == "mae":
        return keras.metrics.MeanAbsoluteError()
    else:
        return ValueError(metric_str + " not a known metric")


def convert_optimizer(optimizer_str, lr):
    if optimizer_str == "adam":
        return keras.optimizers.Adam(lr)
    else:
        return ValueError(optimizer_str + " not a known optimizer")


def load_cfg(cfg_path):
    # cfg_path supplied when running main
    # initialise checkpoint path, set up classes (e.g., MSE loss class instead of mse)
    with open(CFGS_FOLDER + "/" + cfg_path) as stream:
        cfg = yaml.safe_load(stream)
    cfg["path"] = cfg_path.split(".")[0]
    cfg["loss"] = convert_loss(cfg["loss"])
    cfg["metrics"] = [convert_metric(metric) for metric in cfg["metrics"]]
    cfg["optimizer"] = convert_optimizer(cfg["optimizer"], cfg["lr"])
    return cfg
