from typing import List

import numpy as np
from scipy.optimize import least_squares

from helper import normalize, homogeneous


def calculate_homography(obj_points: List[np.ndarray],
                         img_points: List[np.ndarray]
                         ):
    """
    img_vector = H @ obj_vector
    calculate the homography matrix.
    :param obj_points: List[ndarray] 世界坐标系下坐标点
    :param img_points: List[ndarray] 像素坐标下坐标点
    :return: H
    """
    num_point = len(obj_points)
    A = np.zeros([2 * num_point, 9], dtype=np.float64)
    norm_real_mat = normalize(obj_points)
    norm_pixel_mat = normalize(img_points)

    for i in range(num_point):
        Xw, Yw, Zw = obj_points[i]
        upp, vpp = img_points[i]
        norm_up, norm_vp, _ = norm_real_mat @ homogeneous(Xw, Yw)
        norm_upp, norm_vpp, _ = norm_pixel_mat @ homogeneous(upp, vpp)

        A[2 * i, 0] = -norm_up
        A[2 * i, 1] = -norm_vp
        A[2 * i, 2] = -1
        A[2 * i, 6] = norm_upp * norm_up
        A[2 * i, 7] = norm_upp * norm_vp
        A[2 * i, 8] = norm_upp

        A[2 * i + 1, 3] = -norm_up
        A[2 * i + 1, 4] = -norm_vp
        A[2 * i + 1, 5] = -1
        A[2 * i + 1, 6] = norm_vpp * norm_up
        A[2 * i + 1, 7] = norm_vpp * norm_vp
        A[2 * i + 1, 8] = norm_vpp

    u, s, vt = np.linalg.svd(A)

    H = vt[-1].reshape(3, 3)
    H = np.linalg.inv(norm_pixel_mat) @ H @ norm_real_mat
    # H /= H[2, 2]

    return H


def _reproject_error(H: np.ndarray, obj_points: List[np.ndarray], img_points: List[np.ndarray]):
    num_points = len(obj_points)
    res = np.zeros(2 * num_points, dtype=np.float64)
    for i in range(num_points):
        obj_point = obj_points[i]
        pred_point = H @ homogeneous(obj_point).reshape(3, 1)
        pred_point /= pred_point[-1]
        pred_u, pred_v = pred_point[:2]
        res[i * 2] = pred_u - img_points[i][0]
        res[i * 2 + 1] = pred_v - img_points[i][1]

    return res


def jac_h(H: np.ndarray, obj_points: List[np.ndarray], img_points: List[np.ndarray]):
    num_points = len(obj_points)
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = H.flatten()

    jac_H = np.zeros((2 * num_points, H.size))
    for i in range(num_points):
        x = obj_points[2 * i]
        y = obj_points[2 * i + 1]
        u = h11 * x + h12 * y + h13
        v = h21 * x + h22 * y + h23
        w = h31 * x + h32 * y + h33

        jac_H[i * 2, 0] = x / w
        jac_H[i * 2, 1] = y / w
        jac_H[i * 2, 2] = 1 / w
        jac_H[i * 2, 3:6] = 0.
        jac_H[i * 2, 6] = u * x * -1.0 / np.power(w, 2)
        jac_H[i * 2, 7] = u * y * -1.0 / np.power(w, 2)
        jac_H[i * 2, 8] = u * -1.0 / np.power(w, 2)

        jac_H[i * 2 + 1, 0:3] = 0.
        jac_H[i * 2 + 1, 3] = x / w
        jac_H[i * 2 + 1, 4] = y / w
        jac_H[i * 2 + 1, 5] = 1 / w
        jac_H[i * 2 + 1, 6] = v * -1 * x / np.power(w, 2)
        jac_H[i * 2 + 1, 7] = v * -1 * y / np.power(w, 2)
        jac_H[i * 2 + 1, 8] = v * -1 / np.power(w, 2)

    return jac_H


def refine_homography(init_h: np.ndarray, img_points: np.ndarray, obj_points: np.ndarray):
    # initial guess Homography matrix
    homography = init_h.flatten()
    res = least_squares(_reproject_error, homography, jac=jac_h, method="lm", args=(img_points, obj_points))
    refined_h = res.x.reshape(3, 3)
    refined_h /= refined_h[2, 2]

    return refined_h
