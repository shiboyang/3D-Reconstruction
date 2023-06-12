from typing import List

import numpy as np


def _create_v(h: np.ndarray, i: int, j: int):
    return np.array([
        h[0, i] * h[0, j],
        h[1, i] * h[0, j] + h[0, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[1, i] * h[1, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j]
    ])


def calculate_intrinsic_matrix(homograph_list: List[np.ndarray]):
    V = []
    for h in homograph_list:
        v12 = _create_v(h, 0, 1)
        v11 = _create_v(h, 0, 0)
        v22 = _create_v(h, 1, 1)
        V.append(v12)
        V.append(v11 - v22)

    V = np.asarray(V)
    U, S, V = np.linalg.svd(V)
    b11, b12, b13, b22, b23, b33 = V[-1]

    B = np.array([b11, b12, b13,
                  b12, b22, b23,
                  b13, b23, b33]).reshape((3, 3))

    L = np.linalg.cholesky(B)
    # L = cholesky_decompose(B)
    K = np.linalg.inv(L.T)
    K /= K[2, 2]
    K[0, 1] = 0.

    return K
