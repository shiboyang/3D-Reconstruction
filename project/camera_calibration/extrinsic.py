import numpy as np
import cv2


def calculate_extrinsic_vector(K: np.ndarray, H: np.ndarray):
    k_inv = np.linalg.inv(K)
    lamb = 1.0 / np.linalg.norm(k_inv @ H[:, 0])

    r1 = lamb * k_inv @ H[:, 0]
    r2 = lamb * k_inv @ H[:, 1]
    r3 = np.cross(r1, r2)

    t = lamb * k_inv @ H[:, 2]

    R = np.array([r1[0], r2[0], r3[0],
                  r1[1], r2[1], r3[1],
                  r1[2], r2[2], r3[2]]).reshape(3, 3)
    # 由于通过线性的方法求解到的都是近似解 并在列方程过程中也没有严格限制r1 r2 r3的几何性质
    # 通过svd是R为正交矩阵
    u, d, v = np.linalg.svd(R)
    R = u @ v

    R, dRdr = cv2.Rodrigues(R)
    T = t.reshape(3, 1)

    return R, T
