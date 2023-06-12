from typing import List

import cv2
import numpy as np

from helper import homogeneous


def calculate_lens_distortion(img_points: List[List[np.ndarray]], obj_points: List[List[np.ndarray]],
                              intrinsic_matrix: np.ndarray,
                              rotate_vector: List[np.ndarray],
                              translate_vector: List[np.ndarray]):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    u0 = intrinsic_matrix[0, 2]
    v0 = intrinsic_matrix[1, 2]

    num_img = len(img_points)
    num_points = len(obj_points[0])

    # Ax = b
    A = np.zeros((2 * num_img * num_points, 5))
    b = np.zeros((2 * num_img * num_points, 1))

    for i in range(num_img):
        R = rotate_vector[i]
        T = translate_vector[i]
        R, dRdr = cv2.Rodrigues(R)
        extrinsic = np.hstack([R, T])

        for j in range(num_points):
            hom_obj_point = homogeneous(obj_points[i][j])
            Pc = extrinsic @ hom_obj_point
            Xc, Yc, Zc = Pc
            norm_x, norm_y = Xc / Zc, Yc / Zc
            r = np.sqrt(norm_x * norm_x + norm_y * norm_y)

            # 畸变后的像素坐标系下的坐标 -> Z归一化平面上的坐标
            img_point = img_points[i][j]
            upp, vpp = img_point
            up = (upp - u0) / fx
            vp = (vpp - v0) / fy

            row_index = (num_points * i + j) * 2

            A[row_index, 0] = norm_x * r ** 2
            A[row_index, 1] = norm_x * r ** 4
            A[row_index, 2] = norm_x * r ** 6
            A[row_index, 3] = 2 * norm_x * norm_y
            A[row_index, 4] = r ** 2 + 2 * norm_x ** 2
            b[row_index, 0] = up - norm_x

            A[row_index + 1, 0] = norm_y * r ** 2
            A[row_index + 1, 1] = norm_y * r ** 4
            A[row_index + 1, 2] = norm_y * r ** 6
            A[row_index + 1, 3] = r ** 2 + 2 * norm_y ** 2
            A[row_index + 1, 4] = 2 * norm_x * norm_y
            b[row_index + 1, 0] = vp - norm_y

    U, D, Vt = np.linalg.svd(A)
    n = A.shape[-1]
    U = U[:, :n]
    x = Vt.T @ np.linalg.inv(np.diag(D)) @ U.T @ b

    return x
