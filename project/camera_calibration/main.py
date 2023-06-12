import glob
import os

from camclib.calibration import (detect_corner,
                                 create_chessboard_world_coord,
                                 opencv_calibrate, calibrate)

# corner_rows = 6
# corner_clos = 7
# square_size = 60
# pattern_size = (corner_clos, corner_rows)

corner_rows = 6
corner_clos = 9
square_size = 1.
pattern_size = (corner_clos, corner_rows)


def main():
    paths = glob.glob("./data1/*.jpg")
    paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    img_points = []
    obj_points = []
    image_size = None

    obj_point = create_chessboard_world_coord(corner_rows, corner_clos, square_size)
    for path in paths:
        img_point, image_size = detect_corner(path, pattern_size)
        if img_point is not None:
            img_points.append(img_point)
            obj_points.append(obj_point)

    calibrate(img_points, obj_points)
    opencv_calibrate(obj_points, img_points, image_size)


if __name__ == '__main__':
    main()
