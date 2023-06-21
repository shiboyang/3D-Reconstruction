import numpy as np


def calibrate_obj_points(proj_matrix1: np.ndarray,
                         proj_matrix2: np.ndarray,
                         point1: np.ndarray,
                         point2: np.ndarray) -> np.ndarray:
    A = np.zeros([4, 4])
    A[0, :] = proj_matrix1[0, :] - point1[0] * proj_matrix1[2, :]
    A[1, :] = proj_matrix1[1, :] - point1[1] * proj_matrix1[2, :]
    A[2, :] = proj_matrix2[0, :] - point2[0] * proj_matrix2[2, :]
    A[3, :] = proj_matrix2[1, :] - point2[1] * proj_matrix2[2, :]

    u, s, vt = np.linalg.svd(A)
    p = vt[3, :]
    p /= p[-1]

    return p[:-1]
