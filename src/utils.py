"""
Vector-related computing for debugging and testing.
"""

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors
import time


def vec_length(vec):
    return LA.norm(vec)

#tf的循环相关
def circular_correlation(v1, v2):
    # freq_v1 = np.fft.fft(v1)
    freq_v1 = tf.fft(tf.cast(v1, tf.complex64))
    # freq_v2 = np.fft.fft(v2)
    freq_v2 = tf.fft(tf.cast(v2, tf.complex64))
    return tf.real(tf.ifft(tf.multiply(tf.conj(freq_v1), freq_v2)))

#numpy的循环相关
def circular_correlation_np(v1, v2):
    freq_v1 = np.fft.fft(v1)
    freq_v2 = np.fft.fft(v2)
    return np.fft.ifft(np.multiply(freq_v1.conj(), freq_v2)).real

class IndexScore:
    """
    The score of a tail when h and r is given. Or the distance of a vector to the target in kNN sampling.
    It's used in the ranking task to facilitate comparison and sorting.
    Print score as 3 digit precision float.
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        # return "(index: %d, w:%.3f)" % (self.index, self.score)
        return "(%d, %.3f)" % (self.index, self.score)

    def __str__(self):
        return "(index: %d, score:%.3f)" % (self.index, self.score)