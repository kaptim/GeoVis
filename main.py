# needed for quantization-aware training in tensorflow > 2.15
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import sys
from load_cfg import load_cfg
import train

# cfg = load_cfg("NaiveNet96IT_2_1_4_c.yaml")
# example usage: python main.py MobileNetv2_fe_c.yaml fp
# cfg = load_cfg("NaiveNet224IT_3_1_12_c.yaml")
# cfg = load_cfg("ClipNet_fe_c.yaml")


def main():
    cfg = load_cfg(sys.argv[1])
    if sys.argv[2] == "qa":
        train.qa_train(cfg)
    elif sys.argv[2] == "qp":
        train.quantize_post_training(cfg, "")
    elif sys.argv[2] == "fp":
        train.fp_train(cfg)
    elif sys.argv[2] == "ev":
        train.evaluate_model(cfg, "", "", tflite=False)
        train.evaluate_model(cfg, "", "", tflite=True)
    elif sys.argv[2] == "kd":
        cfg_teacher = load_cfg(sys.argv[3])
        print(cfg)
        print(cfg_teacher)
        # train.kd_train(cfg, cfg_teacher)
    else:
        raise ValueError(sys.argv[2] + " not a known function")


if __name__ == "__main__":
    main()
