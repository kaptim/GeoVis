import os

# needed for quantization-aware training in tensorflow > 2.15
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_model_optimization as tfmot

if tf.__version__ >= "2.19.0":
    # keras_hub needs tensorflow >= 2.19
    import keras
    from keras_hub.models import CLIPBackbone

    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

MODELS_PATH = "/home/kaptim/eth/mlmc/project/bottom_up/code/saved_models/"


def wild_test():
    import numpy as np

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    cfg = {}
    cfg["model"] = "naive_net"
    cfg["img_height"] = 28
    cfg["img_width"] = 28
    cfg["dense-1"] = 50
    cfg["task"] = "classification"
    cfg["num_blocks"] = 2
    cfg["block_size"] = 1
    cfg["filters"] = 4
    cfg["maxpool"] = 2
    cfg["averpool"] = 2
    cfg["dropout_conv"] = 0.4
    cfg["dropout_dense"] = 0.4
    cfg["classes"] = np.unique(y_train).tolist()
    cfg["optimizer"] = tf.keras.optimizers.Adam(0.001)
    cfg["loss"] = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    cfg["metrics"] = [tf.keras.metrics.CategoricalAccuracy()]

    x_train = np.reshape(x_train, (60000, 28, 28, 1))
    x_test = np.reshape(x_test, (10000, 28, 28, 1))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(cfg["classes"]))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(cfg["classes"]))

    model = create_model(cfg, "")

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)


class Distiller(tf.keras.Model):
    """
    Custom model that encapsulates knowledge distillation.
    """

    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """
        Args:
            optimizer: Keras optimizer for the student.
            metrics: Keras metrics for the student’s predictions.
            student_loss_fn: Loss function for student outputs (hard labels).
            distillation_loss_fn: Loss function between teacher & student soft predictions.
            alpha: Weight for the student_loss_fn.
            temperature: Temperature for softening logits (teacher & student).
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher in inference mode
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Hard-label loss: student vs. ground truth
            student_loss = self.student_loss_fn(y, student_predictions)

            # Soft targets: apply temperature to teacher & student predictions
            teacher_soft = tf.nn.softmax(teacher_predictions / self.temperature, axis=1)
            student_soft = tf.nn.softmax(student_predictions / self.temperature, axis=1)

            # Distillation loss
            distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft)
            # Multiply by T^2 (common practice from Hinton’s Distillation paper)
            distillation_loss *= self.temperature**2

            # Combine the two losses
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients wrt student
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics (student's accuracy, etc.)
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        # Optionally log individual loss terms
        results.update(
            {"distillation_loss": distillation_loss, "student_loss": student_loss}
        )
        return results

    def test_step(self, data):
        # Unpack data
        x, y = data

        # Student forward pass
        student_predictions = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_predictions)

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


# TODO: feature-based distillation


def naive_conv_block(model, block_size, filters, kernel, strides, padding):
    for i in range(block_size):
        model.add(tf.keras.layers.Conv2D(filters, kernel, strides, padding))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))


def naive_net(cfg):
    # adaptable version of a CNN with regression or classification output
    model = tf.keras.Sequential()
    # specify input dimension
    model.add(tf.keras.Input(shape=(cfg["img_height"], cfg["img_width"], 3)))
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
        model.add(tf.keras.layers.MaxPooling2D((cfg["maxpool"], cfg["maxpool"])))
        model.add(tf.keras.layers.Dropout(cfg["dropout_conv"]))
        filters *= 2

    model.add(tf.keras.layers.AveragePooling2D((cfg["averpool"], cfg["averpool"])))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(cfg["dense-1"], activation="relu"))
    model.add(tf.keras.layers.Dropout(cfg["dropout_dense"]))
    if cfg["task"] == "regression":
        model.add(tf.keras.layers.Dense(len(cfg["targets"])))
    else:
        # classification: tensorflow discourages to add softmax activation function
        model.add(tf.keras.layers.Dense(len(cfg["classes"])))
    return model


def sota_cnn_net(cfg):
    # mobile net v2 with feature extraction / fine tuning capabilities
    img_shape = (cfg["img_height"], cfg["img_width"], 3)
    if cfg["subtask"] == "fine-tuning":
        # fine-tuning: you should pre-train a model on the feature-extraction layers first
        pre_trained_cfg = {key: val for key, val in cfg.items()}
        # change task and path to load the correct model
        pre_trained_cfg["subtask"] = "feature-extraction"
        replace_path_idx = pre_trained_cfg["path"].find("ft")
        pre_trained_cfg["path"] = (
            pre_trained_cfg["path"][:replace_path_idx]
            + "fe"
            + pre_trained_cfg["path"][replace_path_idx + 2 :]
        )
        pretrained_model = load_model(
            pre_trained_cfg, pre_trained_cfg["train_type"], ""
        )
        # unfreeze the whole network or the last block
        pretrained_model.trainable = True
        if cfg["fine_tune_at"] == "last":
            # only keep last "block" of mobile net as trainable
            mobile_net_block = pretrained_model.layers[1]
            for layer in mobile_net_block.layers[:-3]:
                layer.trainable = False
        # whole network: just keep model trainable
        return pretrained_model

    if cfg["model_name"] == "mobile_net_v2":
        application_model = tf.keras.applications.MobileNetV2
    elif cfg["model_name"] == "efficient_net_b0":
        application_model = tf.keras.applications.EfficientNetB0
    else:
        raise ValueError(cfg["model_name"] + " unknown")

    base_model = application_model(
        weights=(None if cfg["subtask"] == "from_scratch" else "imagenet"),
        include_top=False,  # only the feature extraction layers
        input_shape=img_shape,
    )
    if cfg["subtask"] == "feature-extraction":
        # feature extraction: freeze weights
        base_model.trainable = False
    # from-scratch training: just keep base_model trainable

    inputs = tf.keras.Input(shape=img_shape)
    # training=False important in case of batch normalization layers
    x = base_model(
        inputs, training=(True if cfg["subtask"] == "from_scratch" else False)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(cfg["dropout_dense"])(x)
    if cfg["task"] == "regression":
        outputs = tf.keras.layers.Dense(len(cfg["targets"]))(x)
    else:
        # classification: tensorflow discourages to add softmax activation function
        # (numerical instabilities during training)
        outputs = tf.keras.layers.Dense(len(cfg["classes"]))(x)
    return tf.keras.Model(inputs, outputs)


def clip(cfg):
    # CLIP with feature extraction / fine tuning capabilities
    if cfg["subtask"] == "fine-tuning":
        # fine-tuning: you should pre-train a model on the feature-extraction layers first
        pre_trained_cfg = {key: val for key, val in cfg.items()}
        pre_trained_cfg["subtask"] = "feature-extraction"
        replace_path_idx = pre_trained_cfg["path"].find("ft")
        pre_trained_cfg["path"] = (
            pre_trained_cfg["path"][:replace_path_idx]
            + "fe"
            + pre_trained_cfg["path"][replace_path_idx + 2 :]
        )
        pretrained_model = load_model(
            pre_trained_cfg, pre_trained_cfg["train_type"], ""
        )
        vision_encoder = pretrained_model.layers[1]
        vision_encoder.trainable = True
        if cfg["fine_tune_at"] == "last":
            # only keep last "block" of clip as trainable
            for layer in vision_encoder.layers[:-2]:
                layer.trainable = False
        return pretrained_model

    # kerashub: regular keras model
    clip = CLIPBackbone.from_preset(
        cfg["clip_str"],
        load_weights=(False if cfg["subtask"] == "from_scratch" else True),
    )
    vision_encoder = clip.vision_encoder
    vision_pooler = clip.vision_pooler

    if cfg["subtask"] == "feature-extraction":
        # feature extraction: freeze the vision embedding layers
        # pooler: not trainable anyways
        vision_encoder.trainable = False
    # from-scratch training: just keep clip trainable

    inputs = vision_encoder.inputs[0]
    x = vision_encoder(
        inputs, training=(True if cfg["subtask"] == "from_scratch" else False)
    )
    x = vision_pooler(x)
    # architecture inspired by the OSV-5M paper
    # x = keras.layers.Dropout(cfg["dropout_dense"])(x)
    x = keras.layers.Dense(x.shape[1])(x)
    x = keras.layers.GroupNormalization(groups=cfg["group_norm"])(x)
    # x = keras.layers.Dropout(cfg["dropout_dense"])(x)
    x = keras.layers.Dense(cfg["dense-1"])(x)
    x = keras.layers.GroupNormalization(groups=cfg["group_norm"])(x)
    # x = keras.layers.Dropout(cfg["dropout_dense"])(x)
    if cfg["task"] == "regression":
        outputs = keras.layers.Dense(len(cfg["targets"]))(x)
    else:
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
            model = tf.keras.saving.load_model(
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
