import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import numpy as np

EARTH_RADIUS = 6371


def haversine_distance(y_true, y_pred):
    # expects tensor input in (latitude, longitude) format: N x 2

    y_true_radians = tf.map_fn(lambda y: tf.map_fn(lambda x: radians(x), y), y_true)
    y_pred_radians = tf.map_fn(lambda y: tf.map_fn(lambda x: radians(x), y), y_pred)

    distance = np.diag(haversine_distances(y_true_radians, y_pred_radians))

    return EARTH_RADIUS * distance


class HaversineDistance(keras.Metric):
    """Computes the average Haversine distance (distance on a sphere, e.g., the earth)
    between y_true and y_pred
    """

    def __init__(self, name="haversine", **kwargs):
        super().__init__(name=name, **kwargs)
        self.haversine_distance = self.add_variable(
            shape=(), initializer="zeros", name="haversine_distance"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # TODO
        pass
