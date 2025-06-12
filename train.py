import os

# needed for quantization-aware training in tensorflow > 2.15
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot
from load_data import load_dataset
from models import create_model, load_model, MODELS_PATH
from utils import convert_tflite_to_c

RESULTS_PATH = "/home/kaptim/eth/mlmc/project/bottom_up/code/results/"


def train_run(cfg, model, train_type):
    # main training function, input: compiled model
    # train_type = "" if fp, "qat" if qa training
    train_ds, val_ds = load_dataset(cfg, True)

    # best model (based on validation loss) should be saved automatically
    checkpoint_path = MODELS_PATH + cfg["path"] + train_type + ".weights.h5"
    print("Train: Saving weights in " + checkpoint_path)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    csv_logger = keras.callbacks.CSVLogger(
        RESULTS_PATH + cfg["path"] + train_type + ".csv"
    )

    # TODO: lr scheduler?

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=[checkpoint_callback, early_stopping, csv_logger],
    )

    with open(RESULTS_PATH + cfg["path"] + train_type + ".pickle", "wb") as f:
        pickle.dump(history.history, f)


def quantize_post_training(cfg, train_type):
    """Quantize an existing model (full-integer quantization)"""
    quantized_model_path = MODELS_PATH + cfg["path"] + train_type + ".tflite"
    if os.path.isfile(quantized_model_path):
        print(
            "Quantization: " + cfg["path"] + train_type + " has already been quantized"
        )
        return

    model = load_model(cfg, train_type)

    def representative_data_gen():
        # get 100 samples of the train dataset for quantization
        train_ds, val_ds = load_dataset(cfg, True)
        for data in train_ds.unbatch().batch(1).take(100):
            yield [data[0]]

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


def fp_train(cfg):
    # floating-point training (no restrictions on the parameters)
    model = create_model(cfg, "")
    train_run(cfg, model, "")


def qa_train(cfg):
    # quantization-aware training
    # recommended to use a pre-trained fp model
    print("QATrain: Start")
    qa_model = tfmot.quantization.keras.quantize_model(load_model(cfg, ""))
    qa_model.compile(
        optimizer=cfg["optimizer"],
        loss=cfg["loss"],
        metrics=cfg["metrics"],
    )
    train_type = "qat"
    train_run(cfg, qa_model, train_type)
    print("QATrain: Quantize")
    quantize_post_training(cfg, train_type)
    convert_tflite_to_c(cfg, train_type)


def evaluate_model(cfg, train_type, quantized=False):
    # evaluate a checkpointed model on the test set
    test_ds, img_count = load_dataset(cfg, False)
    if not quantized:
        model = load_model(cfg, train_type)
        results = model.evaluate(test_ds)
    else:
        # quantized performance evaluation
        quantized_model_path = MODELS_PATH + cfg["path"] + train_type + ".tflite"
        interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        test_ds = test_ds.unbatch()
        y_pred = np.empty((img_count, len(cfg["targets"])), dtype=int)
        y_true = np.empty((img_count, len(cfg["targets"])), dtype=int)

        for i, data in enumerate(test_ds):
            test_x = data[0]

            input_scale, input_zero_point = input_details["quantization"]
            test_x = test_x / input_scale + input_zero_point

            # expand_dims: create a "batch" of size 1
            test_x = np.expand_dims(test_x, axis=0).astype(input_details["dtype"])
            interpreter.set_tensor(input_details["index"], test_x)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]

            y_pred[i] = output
            y_true[i] = data[1]

        # calculate loss and all metrics
        results = []
        results.append(np.mean(cfg["loss"](y_true, y_pred)).item())
        for metric in cfg["metrics"]:
            results.append(np.mean(metric(y_true, y_pred)).item())

    # TODO: beautify results
    print(results)
