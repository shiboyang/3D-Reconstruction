from typing import List

import cv2
import numpy as np

from triangular__.point_selector import PointSelector
from camclib.homography import calculate_homography
from camclib.extrinsic import calculate_extrinsic_vector


def get_obj_points() -> List[np.ndarray]:
    """
    返回标记点的世界坐标
    """
    points = []

    return points


def get_select_points(img):
    gui = PointSelector(img)
    gui.loop()
    return gui.keypoints


def load_camera_matrix():
    K: np.ndarray

    return K.reshape(3, 3)


def main():
    img1_path = ""
    img2_path = ""
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    img1_points = get_select_points(img1)
    img2_points = get_select_points(img2)

    obj_points = get_obj_points()

    assert len(img1_points) == len(img2_points) == len(obj_points)

    K1 = load_camera_matrix()
    K2 = load_camera_matrix()
    H1 = calculate_homography(obj_points, img2_points)
    H2 = calculate_homography(obj_points, img2_points)
    R1, T1 = calculate_extrinsic_vector(K1, H1)
    R2, T2 = calculate_extrinsic_vector(K2, H2)


