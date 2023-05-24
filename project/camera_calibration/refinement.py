from typing import List

import cv2
import numpy as np
from scipy.optimize import least_squares


def _model(params: np.ndarray, obj_points: List[List[np.ndarray]]):
    fx, fy, u0, v0, k1, k2, k3, p1, p2 = params[:9]
    extrinsic_vector = params[9:]

    num_img = len(obj_points)
    num_points = len(obj_points[0])
    res = np.zeros((num_img * num_points, 2))
    distortion = Distortion(k1, k2, k3, p1, p2)
    for i in range(num_img):
        start = i * 6
        end = (i + 1) * 6
        R = extrinsic_vector[start:end][:3]
        R, dRdr = cv2.Rodrigues(R)
        T = extrinsic_vector[start:end][3:6]
        T = T.reshape(3, 1)
        for j in range(num_points):
            X = obj_points[i][j]
            Pc = R @ X.reshape(3, 1) + T
            Xc, Yc, Zc = Pc.flatten()
            norm_x = Xc / Zc
            norm_y = Yc / Zc
            up, vp = distortion.radial_distortion(norm_x, norm_y) + distortion.tangential_distortion(norm_x, norm_y)

            upp = fx * up + u0
            vpp = fy * vp + v0

            res[i * num_points + j, 0] = upp
            res[i * num_points + j, 1] = vpp

    return res


def _reproject_error(params: np.ndarray, obj_points: List[List[np.ndarray]], img_points: List[List[np.ndarray]]):
    predicted_img_points = _model(params, obj_points)
    img_points = np.asarray(img_points).reshape(-1, 2)

    return (predicted_img_points - img_points).flatten()


def jac_params(params: np.ndarray, obj_points: List[np.ndarray], img_points: List[np.ndarray]):
    num_img = len(img_points)
    num_points = len(obj_points[0])
    # 在param中参数的排列顺序： intrinsic, distortion, extrinsic.
    # intrinsic: fx, fy, u0, v0
    # distortion: k1, k2, k3, p1, p2
    # extrinsic: R[3x3], t1, t2, t3
    intrinsic_vector = params[:4]
    distortion_vector = params[4:9]
    extrinsic_vector = params[9:]
    fx, fy, u0, v0 = intrinsic_vector
    k1, k2, k3, p1, p2 = distortion_vector

    jac_P = np.zeros([2 * num_img * num_points, params.size])
    distortion = Distortion(k1, k2, k3, p1, p2)

    for i in range(num_img):
        start = i * 6
        end = (i + 1) * 6
        R = extrinsic_vector[start: end][:3]
        R, dRdr = cv2.Rodrigues(R)
        T = extrinsic_vector[start: end][3:6]
        T = T.reshape(3, 1)

        for j in range(num_points):
            obj_point = obj_points[i][j]
            img_point = img_points[i][j]
            row_index = (i * num_points + j) * 2
            point_index = 9 + i * 6

            X = obj_point.reshape(3, 1)
            Pc = R @ X + T
            Xc, Yc, Zc = Pc.flatten()
            norm_x = Xc / Zc
            norm_y = Yc / Zc
            up, vp = distortion.radial_distortion(norm_x, norm_y) + distortion.tangential_distortion(norm_x, norm_y)

            # intrinsic
            jac_P[row_index, 0] = up
            jac_P[row_index, 1] = 0
            jac_P[row_index, 2] = 1
            jac_P[row_index, 3] = 0
            # distortion
            # radial distortion
            jac_P[row_index, 4] = fx * distortion.dk1(norm_x, norm_y)
            jac_P[row_index, 5] = fx * distortion.dk2(norm_x, norm_y)
            jac_P[row_index, 6] = fx * distortion.dk3(norm_x, norm_y)
            # tangential distortion
            jac_P[row_index, 7] = fx * distortion.dupdp1(norm_x, norm_y)
            jac_P[row_index, 8] = fx * distortion.dupdp2(norm_x, norm_y)

            # extrinsic R d(u'')/d(R) = d(u'')/d(Pc) * d(Pc)/d(R)
            # du''/dPc = [du''/dXc, du''/dYc, du''/dZc]

            # du''/dXc = du''/dup * dup/dnorm_x * dnorm_x/dXc
            dUppdXc = fx * distortion.dx(norm_x, norm_y) * 1.0 / Zc
            # du''/dYc = du''/dup * dup/dnorm_x * dnorm_x/dYc
            dUppdYc = 0
            # du''/dZc = du''/dup * dup/dnorm_x * dnorm_x/dZc
            dUppdZc = fx * distortion.dx(norm_x, norm_y) * -Xc / Zc ** 2

            dUppdPc = np.array([dUppdXc, dUppdYc, dUppdZc]).reshape(1, 3)
            # dPc/dR_ij = [0,
            #              0,
            #              ...,
            #              x_j, <- ith
            #              0]
            # du''/dR_ij = {du''/dPc}_i * x_j
            dUppdR = dUppdPc.T @ X.T
            jac_P[row_index, point_index:point_index + 6][:3] = dUppdR.reshape(1, 9) @ dRdr.T

            # extrinsic T du''/dT = du''/dPc * dPc/dT
            # dPc/dT_i = [0,
            #             1, <-ith
            #             0,
            #             ...,
            #             0]
            # du''/dT_i = {du''/dPc}_i
            jac_P[row_index, point_index:point_index + 6][3:] = dUppdPc.flatten()

            # -----------------

            # intrinsic
            jac_P[row_index + 1, 0] = 0  # dv''/dfx
            jac_P[row_index + 1, 1] = vp  # dv''/dfy
            jac_P[row_index + 1, 2] = 0  # dv''/du0
            jac_P[row_index + 1, 3] = 1  # dv''/dv0
            # distortion k1 k2 k3 p1 p2
            jac_P[row_index + 1, 4] = fy * distortion.dk1(norm_x, norm_y)  # dv''/dk1 = dv''/dvp * dvp/dk1
            jac_P[row_index + 1, 5] = fy * distortion.dk2(norm_x, norm_y)  # dv''/dk2 = dv''/dvp * dvp/dk2
            jac_P[row_index + 1, 6] = fy * distortion.dk3(norm_x, norm_y)  # dv''/dk3 = dv''/dvp * dvp/dk3
            jac_P[row_index + 1, 7] = fy * distortion.dvpdp1(norm_x, norm_y)  # dv''/dp1 = dv''/dvp * dvp/dp1
            jac_P[row_index + 1, 8] = fy * distortion.dvpdp2(norm_x, norm_y)  # dv''/dp2 = dv''/dvp * dvp/dp2
            # extrinsic R[3x3] T[3X1]
            dVppdXc = 0
            # dv''/dYc = dv''/dvp * dvp/dnorm_y * dnorm_y/dYc
            dVppdYc = fy * distortion.dy(norm_x, norm_y) * 1.0 / Zc
            # dv''/dZc = dv''/dvp * dvp/dnorm_y * dnorm_y/dZc
            dVppdZc = fy * distortion.dy(norm_x, norm_y) * -Yc / Zc ** 2
            dVppdPc = np.array([dVppdXc, dVppdYc, dVppdZc]).reshape(1, 3)
            # dv''/dR = dv''/dPc * dPc/dR
            dVppdR = dVppdPc.T @ X.T
            jac_P[row_index + 1, point_index:point_index + 6][:3] = dVppdR.reshape(1, 9) @ dRdr.T
            # dv''/dT = dv''/dPc * dPc/dT
            jac_P[row_index + 1, point_index:point_index + 6][3:] = dVppdPc.flatten()

    return jac_P


class Distortion:
    def __init__(self, k1: float, k2: float, k3: float, p1: float, p2: float):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

    def radial_distortion(self, x: float, y: float):
        r = self._r(x, y)
        x = x * (1 + self.k1 * r ** 2 + self.k2 * r ** 4 + self.k3 * r ** 6)
        y = y * (1 + self.k1 * r ** 2 + self.k2 * r ** 4 + self.k3 * r ** 6)
        return np.array([x, y])

    def tangential_distortion(self, x: float, y: float):
        r = self._r(x, y)
        x = 2 * self.p1 * x * y + self.p2 * (r ** 2 + 2 * x ** 2)
        y = self.p1 * (r ** 2 + 2 * y ** 2) + 2 * self.p2 * x * y

        return np.array([x, y])

    @staticmethod
    def _r(x: float, y: float):
        return np.sqrt(x ** 2 + y ** 2)

    def dk1(self, x: float, y: float):
        r = self._r(x, y)
        return x * r ** 2

    def dk2(self, x: float, y: float):
        r = self._r(x, y)
        return x * r ** 4

    def dk3(self, x: float, y: float):
        r = self._r(x, y)
        return x * r ** 6

    def dupdp1(self, x: float, y: float):
        return 2 * x * y

    def dupdp2(self, x: float, y: float):
        r = self._r(x, y)
        return r ** 2 + 2 * x ** 2

    def dvpdp1(self, x: float, y: float):
        r = self._r(x, y)
        return r ** 2 + 2 * y ** 2

    def dvpdp2(self, x: float, y: float):
        return 2 * x * y

    def L(self, r):
        return 1 + self.k1 * r ** 2 + self.k2 * r ** 4 + self.k3 * r ** 6

    def dx(self, x: float, y: float):
        # L = 1 + k1 * r^2 + k2 * r^4 + k3 * r^6
        # r^2 = x^2 + y^2
        # qx = 2 * p1 * x * y + p2 * (r^2 + 2 * x * y)
        # up = x * L + q
        # dup/dx = L + x * dL/dx + dqx/dx
        r = self._r(x, y)
        L = self.L(r)
        dLdx = x * (self.k1 * 2 + self.k2 * 4 * r ** 2 + self.k3 * 6 * r ** 4)
        dqxdx = 2 * self.p1 * y + 6 * x * self.p2
        return L + x * dLdx + dqxdx

    def dy(self, x: float, y: float):
        # dvp/dy = L + y * dL/dv + dqv/dv
        r = self._r(x, y)
        L = self.L(r)
        dLdy = y * (self.k1 * 2 + self.k2 * 4 * r ** 2 + self.k3 * 6 * r ** 4)
        dqydy = 6 * y * self.p1 + 2 * self.p2 * x
        return L + y * dLdy + dqydy


def refine_all_params(img_points: List[np.ndarray], obj_points: List[np.ndarray],
                      intrinsic: np.ndarray,
                      rotate_vector: List[np.ndarray],
                      translate_vector: List[np.ndarray],
                      k1: float, k2: float, k3: float,
                      p1: float, p2: float):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    u0 = intrinsic[0, 2]
    v0 = intrinsic[1, 2]
    params = np.array([fx, fy, u0, v0, k1, k2, k3, p1, p2], dtype=np.float64)

    for r, t in zip(rotate_vector, translate_vector):
        params = np.append(params, r.flatten())
        params = np.append(params, t.flatten())

    res = least_squares(_reproject_error, params, jac=jac_params, method="lm", args=(obj_points, img_points))

    return res.x.flatten()
