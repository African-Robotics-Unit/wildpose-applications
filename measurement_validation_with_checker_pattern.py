import os
import sys


ECAL_FOLDER = ''
LIDAR_FRAME_FPATHS = [

]


def main():
    # load the data
    pcd_fpaths = [
        os.path.join(ECAL_FOLDER, 'lidar', x)
        for x in LIDAR_FRAME_FPATHS
    ]

    # load the camera parameters

    # get the checker pattern points from the image

    # project the 3D points onto the image

    # get 3D point indices corresponding with checker pattern

    # show the 3D lengths
    pass


if __name__ == '__main__':
    main()
