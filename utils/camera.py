import numpy as np


def make_intrinsic_mat(fx, fy, cx, cy):
    intrinsic_mat = np.eye(4)
    intrinsic_mat[0, 0] = fx
    intrinsic_mat[0, 2] = cx
    intrinsic_mat[1, 1] = fy
    intrinsic_mat[1, 2] = cy

    return intrinsic_mat


def make_extrinsic_mat(rot_mat, translation):
    extrinsic_mat = np.eye(4)
    extrinsic_mat[:3, :3] = rot_mat
    extrinsic_mat[:-1, -1] = translation
    extrinsic_mat[-1, -1] = 1

    return extrinsic_mat
