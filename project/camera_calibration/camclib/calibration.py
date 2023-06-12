from typing import List, Tuple

import cv2
import numpy as np

from distortion import calculate_lens_distortion
from extrinsic import calculate_extrinsic_vector
from homography import calculate_homography
from intrinsic import calculate_intrinsic_matrix
from refinement import refine_all_params


def create_chessboard_world_coord(rows: int, cols: int, square_size: float) -> List[np.ndarray]:
    points = np.zeros([rows, cols, 3], dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            x, y, z = c, r, 0
            points[r, c, :] = x, y, z

    points *= square_size
    points = points.reshape(-1, 3)
    obj_points = []
    for p in points:
        obj_points.append(np.array(p))
    return obj_points


def detect_corner(image_path: str, pattern_size, show_corner=False) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    img_points = []
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray_image.shape
    ret, corners = cv2.findChessboardCorners(gray_image, pattern_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ret:
        corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

    # 绘制棋盘格角点
    if show_corner:
        cv2.drawChessboardCorners(image, pattern_size, corners, True)
        cv2.imshow(image_path, image)
        cv2.waitKey(0)

    if ret:
        corners = corners.reshape(-1, 2)
        for corner in corners:
            img_points.append(corner)

    return img_points, image_size


# todo
# def cholesky_decompose(B: np.ndarray):
#     L = np.zeros_like(B)
#     h, w = L.shape
#     for i in range(h):
#         for j in range(w):
#             if i > j:
#                 L[i, j] = (B[i, j] - np.sum(L[i, :] * L[j, :])) / L[j, j]
#             else:
#                 np.sqrt(B[i, i] - np.sum(L[i, :] * L[i, :]))
#
#     return L


def calibrate(img_points: List[List[np.ndarray]],
              obj_points: List[List[np.ndarray]]
              ):
    rotate_vector = []
    translate_vector = []
    homography_matrices = []
    num_img = len(img_points)
    for i in range(num_img):
        h = calculate_homography(obj_points[i], img_points[i])
        homography_matrices.append(h)

    K = calculate_intrinsic_matrix(homography_matrices)

    for h in homography_matrices:
        R, T = calculate_extrinsic_vector(K, h)
        rotate_vector.append(R)
        translate_vector.append(T)

    k1, k2, k3, p1, p2 = calculate_lens_distortion(img_points, obj_points, K, rotate_vector, translate_vector)

    params = refine_all_params(img_points, obj_points, K, rotate_vector, translate_vector, k1, k2, k3, p1, p2)

    fx, fy, u0, v0 = params[:4]
    k1, k2, k3, p1, p2 = params[4:9]
    print(fx, fy, u0, v0)
    print(k1, k2, k3, p1, p2)


def opencv_calibrate(obj_points: List[List[np.ndarray]],
                     img_points: List[List[np.ndarray]],
                     image_size):
    # obj_points = np.asarray(obj_points, dtype=np.float64)
    # img_points = np.asarray(img_points, dtype=np.float64)

    num_img = len(img_points)
    obj_point_list = []
    img_point_list = []
    for i in range(num_img):
        obj_point_list.append(np.asarray(obj_points[i]))
        img_point_list.append(np.asarray(img_points[i]))

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(obj_point_list, img_point_list, image_size,
                                                                         None,
                                                                         None)
    print("ret: {}".format(retval))
    print("intrinsic matrix: \n {}".format(cameraMatrix))
    # in the form of (k_1, k_2, p_1, p_2, k_3)
    print("distortion cofficients: \n {}".format(distCoeffs))
    print("rotation vectors: \n {}".format(rvecs))

    print("translation vectors: \n {}".format(tvecs))
