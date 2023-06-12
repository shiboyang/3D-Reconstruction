from typing import List, Union

import cv2
import numpy as np


def stddiff(data: np.ndarray) -> float:
    mean = data.mean()
    n = data.shape[0]
    return np.sqrt(np.square(data - mean).sum() / n)


def normalize(points: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(points, list):
        points = np.asarray(points)

    cx = points[:, 0].mean()
    cy = points[:, 1].mean()

    stdx = stddiff(points[:, 0])
    stdy = stddiff(points[:, 1])

    scale_x = np.sqrt(2.0) / stdx
    scale_y = np.sqrt(2.0) / stdy

    T = np.array([scale_x, 0, -scale_x * cx,
                  0, scale_y, -scale_y * cy,
                  0, 0, 1]).reshape(3, 3)
    return T


def homogeneous(*args) -> np.ndarray:
    if len(args) == 1:
        array = np.asarray(args[0])
    else:
        array = np.asarray(args)
    array = np.append(array.flatten(), 1)
    return array


def check_homograph(H: np.ndarray, pts1: List[np.ndarray], pts2: List[np.ndarray]):
    h_pts1 = cv2.convertPointsToHomogeneous(pts1)
    h_pts2 = h_pts1 @ H
    cal_pts2 = cv2.convertPointsFromHomogeneous(h_pts2)
    for p1, p2 in zip(cal_pts2, pts2):
        print(p1, "  ", p2, " ", p1 / p2)
