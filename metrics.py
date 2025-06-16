import tensorflow as tf
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import numpy as np

EARTH_RADIUS = 6371


def haversine_distance(y_true, y_pred):
    # expects tensor input in (latitude, longitude) format: N x 2
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    y_true_radians = tf.map_fn(lambda y: tf.map_fn(lambda x: radians(x), y), y_true)
    y_pred_radians = tf.map_fn(lambda y: tf.map_fn(lambda x: radians(x), y), y_pred)

    distance = np.diag(haversine_distances(y_true_radians, y_pred_radians))

    return tf.Variable(EARTH_RADIUS * distance)


class HaversineDistance(tf.keras.metrics.Mean):
    """Computes the average Haversine distance (distance on a sphere, e.g., the earth)
    between y_true and y_pred
    """

    def __init__(self, name="haversine", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self._fn = haversine_distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        print("test")
        values = self._fn(y_true, y_pred)
        super().update_state(values, sample_weight=sample_weight)


h = HaversineDistance()
