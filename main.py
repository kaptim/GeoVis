import os

# needed for quantization-aware training in tensorflow > 2.15
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
from load_cfg import load_cfg
import train

# cfg = load_cfg("MobileNetv2_fe_c.yaml")
# example usage: python main.py MobileNetv2_fe_c.yaml fp
# cfg = load_cfg("NaiveRegNet2_1_4_04_04.yaml")
# cfg = load_cfg("ClipNet_fe_c.yaml")


def main():
    cfg = load_cfg(sys.argv[1])
    if sys.argv[2] == "qa":
        train.qa_train(cfg)
    elif sys.argv[2] == "qp":
        train.quantize_post_training(cfg)
    elif sys.argv[2] == "fp":
        train.fp_train(cfg)
    else:
        raise ValueError(sys.argv[2] + " not a known function")


if __name__ == "__main__":
    main()
