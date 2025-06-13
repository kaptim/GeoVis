import os

# needed for quantization-aware training in tensorflow > 2.15
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import model_compression_toolkit as mct
from load_data import load_dataset
from models import load_model, MODELS_PATH

CFG = ""


def representative_data_gen():
    # get 200 samples of the train dataset for quantization
    train_ds, val_ds = load_dataset(CFG, True)
    for data in train_ds.unbatch().batch(1).take(200):
        yield [data[0]]


def quantize_post_training(cfg, train_type):
    """Quantize an existing model (full-integer quantization)
    train_type: "" if fp, "qat" if qa training"""
    quantized_model_path = MODELS_PATH + cfg["path"] + train_type + ".tflite"
    if os.path.isfile(quantized_model_path):
        print(
            "Quantization: " + cfg["path"] + train_type + " has already been quantized"
        )
        return

    model = load_model(cfg, train_type, "")
    # change global variable CFG for representative_data_gen to work
    global CFG
    CFG = cfg

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()

    with open(quantized_model_path, "wb") as f:
        f.write(tflite_quant_model)

    print("Quantization: " + cfg["path"] + train_type + " was quantized")


def quantize_post_training_mct(cfg, train_type):
    """Quantize an existing model (full-integer quantization) using Sony's model compression toolkit
    train_type: "" if fp, "qat" if qa training"""
    # have to use .keras
    quantized_model_path = MODELS_PATH + cfg["path"] + "mct" + train_type + ".keras"
    if os.path.isfile(quantized_model_path):
        print(
            "Quantization: "
            + cfg["path"]
            + "mct"
            + train_type
            + " has already been quantized"
        )
        return

    model = load_model(cfg, train_type, "")
    # change global variable CFG for representative_data_gen to work
    global CFG
    CFG = cfg

    # get a FrameworkQuantizationCapabilities object that models the hardware for the quantized model inference.
    # default platform: IMX500
    target_platform_cap = mct.get_target_platform_capabilities("tensorflow", "default")
    quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(
        in_model=model,
        representative_data_gen=representative_data_gen,
        target_platform_capabilities=target_platform_cap,
    )
    mct.exporter.keras_export_model(
        model=quantized_model, save_model_path=quantized_model_path
    )
