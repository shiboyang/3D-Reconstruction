#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

class SingleCamera {
public:
    SingleCamera(Eigen::MatrixXf world_coor, Eigen::MatrixXf pixel_coor, int n)
            : world_coor(world_coor), pixel_coor(pixel_coor), point_num(n),
              P(Eigen::MatrixXf::Zero(n, 12)), M(Eigen::MatrixXf::Zero(3, 4)),
              A(Eigen::MatrixXf::Zero(3, 3)), b(Eigen::MatrixXf::Zero(3, 1)),
              K(Eigen::MatrixXf::Zero(3, 3)), R(Eigen::MatrixXf::Zero(3, 3)),
              t(Eigen::MatrixXf::Zero(3, 1)) {}

    void composeP();

    void svdP();

    void workIntrinsicAndExtrinsic();

    void selfcheck(const Eigen::MatrixXf &w_check, const Eigen::MatrixXf &c_check);

private:
    Eigen::MatrixXf world_coor;
    Eigen::MatrixXf pixel_coor;
    int point_num;

    // 变量都是与课程PPT相对应的
    Eigen::MatrixXf P;
    Eigen::MatrixXf M;
    Eigen::MatrixXf A;
    Eigen::MatrixXf b;
    Eigen::MatrixXf K;
    Eigen::MatrixXf R;
    Eigen::MatrixXf t;
};

void SingleCamera::composeP() {
    // homework1: 根据输入的二维点和三维点，构造P矩阵
    Eigen::RowVectorXf row(12), p(4);
    float u, v;
    for (int i = 0; i < point_num; i++) {
        // 一次生成两个方程系数
        if (i % 2 == 0) {
            p = world_coor.row(i);
            u = pixel_coor(i, 0);
            v = pixel_coor(i, 1);
            row << p,
                    Eigen::RowVectorXf::Zero(1, 4),
                    -u * p;
            P.row(i) = row;
            row << Eigen::RowVectorXf::Zero(1, 4),
                    p,
                    -v * p;
            P.row(i + 1) = row;
        }
    }

//    std::cout << "P:\n" << P << std::endl;

}

void SingleCamera::svdP() {
    // homework2: 根据P矩阵求解M矩阵和A、b矩阵
    Eigen::JacobiSVD<Eigen::MatrixXf> svd;
    svd.compute(P, Eigen::ComputeThinU | Eigen::ComputeThinV);
    M = svd.matrixV().col(svd.matrixV().cols() - 1);
    M.resize(4, 3);
    M.transposeInPlace();
    A = M.leftCols(3);   //[3, 3]
    b = M.rightCols(1);  //[3, 1]
}

void SingleCamera::workIntrinsicAndExtrinsic() {
    // homework3: 求解相机的内参和外参
    auto size = A.rows();
    Eigen::RowVector3f a1(A.row(0)), a2(A.row(1)), a3(A.row(2));

    float roh = -1 / a3.norm();
    auto cx = pow(roh, 2) * (a1.dot(a3));
    auto cy = pow(roh, 2) * (a2.dot(a3));
    auto a1xa3 = a1.cross(a3);
    auto a2xa3 = a2.cross(a3);
    auto cos_theta = -1 * a1xa3.dot(a2xa3) / (a1xa3.norm() * a2xa3.norm());
    auto theta = acos(cos_theta) * 180 / M_PI;
    auto alpha = pow(roh, 2) * a1xa3.norm() * sin(theta);
    auto beta = pow(roh, 2) * a2xa3.norm() * sin(theta);

    auto r1 = a2xa3 / a2xa3.norm();
    auto r3 = a3 / a3.norm();
    auto r2 = r3.cross(r1);


    K(0, 0) = alpha;
    K(0, 1) = -alpha * cos(theta) / sin(theta);
    K(0, 2) = cx;
    K(1, 1) = beta / sin(theta);
    K(1, 2) = cy;
    K(2, 2) = 1;

    R.row(0) = r1;
    R.row(1) = r2;
    R.row(2) = r3;

    t = roh * K.inverse() * b;

    std::cout << "K is " << std::endl << K << std::endl;
    std::cout << "R is " << std::endl << R << std::endl;
    std::cout << "t is " << std::endl << t.transpose() << std::endl;
}

void SingleCamera::selfcheck(const Eigen::MatrixXf &w_check, const Eigen::MatrixXf &c_check) {
    float average_err = DBL_MAX;
    // homework4: 根据homework3求解得到的相机的参数，使用测试点进行验证，计算误差
    Eigen::MatrixXf m(3, 4);
    m << R, t;

    auto res = world_coor * (K * m).transpose();

    std::cout << "res: \n" << res << std::endl;
    std::cout << "pixel_coor: \n" << pixel_coor << std::endl;


    std::cout << "The average error is " << average_err << "," << std::endl;
    if (average_err > 0.1) {
        std::cout << "which is more than 0.1" << std::endl;
    } else {
        std::cout << "which is smaller than 0.1, the M is acceptable" << std::endl;
    }
}


int main(int argc, char **argv) {

    Eigen::MatrixXf w_xz(4, 4);
    w_xz << 8, 0, 9, 1,
            8, 0, 1, 1,
            6, 0, 1, 1,
            6, 0, 9, 1;

    Eigen::MatrixXf w_xy(4, 4);
    w_xy << 5, 1, 0, 1,
            5, 9, 0, 1,
            4, 9, 0, 1,
            4, 1, 0, 1;

    Eigen::MatrixXf w_yz(4, 4);
    w_yz << 0, 4, 7, 1,
            0, 4, 3, 1,
            0, 8, 3, 1,
            0, 8, 7, 1;

    Eigen::MatrixXf w_coor(12, 4);
    w_coor << w_xz,
            w_xy,
            w_yz;

    Eigen::MatrixXf c_xz(4, 2);
    c_xz << 275, 142,
            312, 454,
            382, 436,
            357, 134;

    Eigen::MatrixXf c_xy(4, 2);
    c_xy << 432, 473,
            612, 623,
            647, 606,
            464, 465;

    Eigen::MatrixXf c_yz(4, 2);
    c_yz << 654, 216,
            644, 368,
            761, 420,
            781, 246;

    Eigen::MatrixXf c_coor(12, 2);
    c_coor << c_xz,
            c_xy,
            c_yz;

    Eigen::MatrixXf w_check(5, 4);
    w_check << 6, 0, 5, 1,
            3, 3, 0, 1,
            0, 4, 0, 1,
            0, 4, 4, 1,
            0, 0, 7, 1;

    Eigen::MatrixXf c_check(5, 2);
    c_check << 369, 297,
            531, 484,
            640, 468,
            646, 333,
            556, 194;

    SingleCamera aCamera = SingleCamera(w_coor, c_coor, 12);  // 12 points in total are used
    aCamera.composeP();
    aCamera.svdP();
    aCamera.workIntrinsicAndExtrinsic();
    aCamera.selfcheck(w_check, c_check);  // test 5 points and verify M


    return 0;
}