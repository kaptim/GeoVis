import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # or "tensorflow" or "torch"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

# needed for quantization-aware training in tensorflow > 2.15
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import keras
from keras_hub.models import CLIPBackbone

MODELS_PATH = "/home/kaptim/eth/mlmc/project/bottom_up/code/saved_models/"


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


def mobile_net_v2_fe(cfg):
    # mobile net v2 with feature extraction
    img_shape = (cfg["img_height"], cfg["img_width"], 3)
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,  # only the feature extraction layers
        input_shape=img_shape,
    )
    # freeze weights of the feature extractor
    base_model.trainable = False
    # create model
    inputs = tf.keras.Input(shape=img_shape)
    # training=False important in case of batch normalization layers
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(cfg["dropout_dense"])(x)
    if cfg["task"] == "regression":
        outputs = tf.keras.layers.Dense(len(cfg["targets"]))(x)
    else:
        # classification: tensorflow discourages to add softmax activation function
        # (numerical instabilities during training)
        outputs = tf.keras.layers.Dense(len(cfg["classes"]))(x)
    return tf.keras.Model(inputs, outputs)


def clip_fe(cfg):
    # CLIP using feature extraction
    # TODO: lora?
    # kerashub: regular keras model
    clip = CLIPBackbone.from_preset("clip_vit_b_32_laion2b_s34b_b79k")
    vision_encoder = clip.vision_encoder
    vision_pooler = clip.vision_pooler
    vision_projection = clip.vision_projection

    # freeze the vision embedding layers
    vision_encoder.trainable = False
    vision_pooler.trainable = False
    vision_projection.trainable = False

    inputs = clip.inputs[0]
    x = vision_encoder(inputs, training=False)
    x = vision_pooler(x)
    x = vision_projection(x)

    x = keras.layers.Dropout(cfg["dropout_dense"])(x)
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
