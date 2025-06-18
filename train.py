# needed for quantization-aware training in tensorflow > 2.15
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

if tf.__version__ >= "2.19.0":
    # keras_hub needs tensorflow >= 2.19
    from keras_hub.models import CLIPBackbone, CLIPTokenizer
from load_data import load_dataset
from models import create_model, load_model, Distiller, MODELS_PATH
from utils import convert_tflite_to_c
from quantization import quantize_post_training

RESULTS_PATH = "/home/kaptim/eth/mlmc/project/bottom_up/code/results/"


def train_run(cfg, model, train_type):
    # main training function, input: compiled model
    # train_type = "" if fp, "qat" if qa training
    train_ds, val_ds = load_dataset(cfg, True)

    # best model (based on validation loss) should be saved automatically
    checkpoint_path = (
        MODELS_PATH + "weights/" + cfg["path"] + train_type + ".weights.h5"
    )
    print("Train: Saving weights in " + checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=(
            "val_loss" if cfg["task"] == "regression" else "val_categorical_accuracy"
        ),
        mode=("min" if cfg["task"] == "regression" else "max"),
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    # TODO: learning rate scheduler?
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=(
            "val_loss" if cfg["task"] == "regression" else "val_categorical_accuracy"
        ),
        patience=40,
    )

    # save training and validation results
    csv_logger = tf.keras.callbacks.CSVLogger(
        RESULTS_PATH + cfg["path"] + train_type + ".csv"
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=[checkpoint_callback, csv_logger, early_stopping],
    )


def fp_train(cfg):
    # floating-point training (no restrictions on the parameters)
    model = create_model(cfg, "")
    train_run(cfg, model, "")


def qa_train(cfg):
    # quantization-aware training
    # recommended to use a pre-trained fp model
    print("QATrain: Start")
    qa_model = tfmot.quantization.keras.quantize_model(load_model(cfg, "", ""))
    qa_model.compile(
        optimizer=cfg["optimizer"],
        loss=cfg["loss"],
        metrics=cfg["metrics"],
    )
    name = "qat"
    train_run(cfg, qa_model, name)
    print("QATrain: Quantize")
    quantize_post_training(cfg, name)


def kd_train(cfg_student, cfg_teacher, kd_alpha=0.4, kd_temp=2):
    # kd_alpha: knowledge distillation alpha value, lower value => distillation loss more important
    # kd_temp: knowledge distillation temperature value, smooths the probability distributions
    print(
        "Knowledge distillation, student: "
        + cfg_student["path"]
        + ", teacher: "
        + cfg_teacher["path"]
    )
    # train using the teacher's preprocessor
    cfg_student["preprocessor"] = cfg_teacher["preprocessor"]
    distiller = Distiller(
        create_model(cfg_student, ""), load_model(cfg_teacher, "", "")
    )
    distiller.compile(
        optimizer=cfg_student["optimizer"],
        metrics=cfg_student["metrics"],
        student_loss_fn=cfg_student["loss"],
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=kd_alpha,
        temperature=kd_temp,
    )
    train_run(cfg_student, distiller, "")


def test_run(cfg, train_type, mct, tflite):
    # load and run model on the test set
    test_ds, img_count = load_dataset(cfg, False)
    data_t = int if tflite else np.float32
    if cfg["task"] == "regression":
        y_pred = np.empty((img_count, len(cfg["targets"])), dtype=data_t)
        y_true = np.empty((img_count, len(cfg["targets"])), dtype=data_t)
    else:
        y_pred = np.empty((img_count, len(cfg["classes"])), dtype=data_t)
        y_true = np.empty((img_count, len(cfg["classes"])), dtype=data_t)
    if not tflite:
        model = load_model(cfg, train_type, mct)
        i = 0
        for batch in test_ds:
            test_x = batch[0]
            batch_size = batch[0].shape[0]
            y_true[i : (i + batch_size)] = batch[1].numpy()
            y_pred[i : (i + batch_size)] = model.predict(test_x)
            i += batch_size
    else:
        # quantized performance evaluation (.tflite models)
        quantized_model_path = MODELS_PATH + cfg["path"] + train_type + ".tflite"
        interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        test_ds = test_ds.unbatch()
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

    return y_true, y_pred


def evaluate_model(cfg, train_type, mct, tflite=False):
    # evaluate a checkpointed model on the test set
    if not tflite:
        test_ds, img_count = load_dataset(cfg, False)
        model = load_model(cfg, train_type, mct)
        results = model.evaluate(test_ds)
    else:
        y_true, y_pred = test_run(cfg, train_type, mct, True)
        # compute all metrics
        results = []
        for metric in cfg["metrics"]:
            results.append(np.mean(metric(y_true, y_pred)).item())

    if not os.path.exists(RESULTS_PATH + "test_results.csv"):
        with open(RESULTS_PATH + "test_results.csv", "w") as fd:
            fd.write("Name,QAT,MCT,TFLITE,Accuracy\n")
    # append test results
    with open(RESULTS_PATH + "test_results.csv", "a") as fd:
        fd.write(
            ",".join([cfg["path"], train_type, mct, str(tflite), str(results[-1])])
            + "\n"
        )


def evaluate_clip(cfg):
    # function to evaluate the performance of raw CLIP
    # classification only since CLIP is trained to associate images with words
    test_ds, img_count = load_dataset(cfg, False)
    test_ds = test_ds.unbatch().batch(1)

    clip = CLIPBackbone.from_preset(
        cfg["clip_str"],
        load_weights=True,
    )
    tokenizer = CLIPTokenizer.from_preset(cfg["clip_str"], sequence_length=20)
    tokens = tokenizer.tokenize(
        ["A Street View photo from " + c for c in cfg["classes"].tolist()]
    )

    correct = 0
    for i, data in enumerate(test_ds):
        output = clip(
            {
                "images": data[0],
                "token_ids": tokens,
            }
        )
        if (
            tf.argmax(output["vision_logits"], axis=1)[0].numpy()
            == tf.argmax(data[1], axis=1)[0].numpy()
        ):
            correct += 1

    print(correct / img_count)
