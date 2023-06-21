import numpy as np


def hnormalized(v: np.ndarray):
    v /= v[-1]
    return v[:-1]
