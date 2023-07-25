import cv2
import numpy as np
import open3d as o3d
import pickle
import rigid_body_motion as rbm
import quaternion
import glob
import os
import pdb
import json
from tqdm import tqdm
from pyquaternion import Quaternion

COLORS = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1,
        "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1,
        "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0,
        "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0,
        "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0,
        "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0,
        "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0,
        "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0,
        "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0,
        "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0,
        "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0,
        "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0,
        "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0,
        "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0,
        "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0,
        "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0,
        "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0,
        "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

colors_indice = [0, 5, 10, 15, 20, 25, 30, 35, 13, 45, 50, 55, 60, 65, 70]


def make_intrinsic(fx, fy, cx, cy):

    intrinsic_mat = np.eye(4)
    intrinsic_mat[0, 0] = fx
    intrinsic_mat[0, 2] = cx
    intrinsic_mat[1, 1] = fy
    intrinsic_mat[1, 2] = cy

    return intrinsic_mat


def make_extrinsic(rot_mat, translation):

    extrinsic_mat = np.eye(4)
    extrinsic_mat[:3, :3] = rot_mat
    extrinsic_mat[:-1, -1] = translation
    extrinsic_mat[-1, -1] = 1

    return extrinsic_mat


def lidar2cam_projection(pcd, extrinsic):
    velo = np.insert(pcd, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    pcd_in_cam = extrinsic.dot(velo)

    return pcd_in_cam


def cam2image_projection(pcd_in_cam, intrinsic):
    pcd_in_image = intrinsic.dot(pcd_in_cam)
    pcd_in_image[:2] /= pcd_in_image[2, :]

    return pcd_in_image


def get_3D_by_2Dloc(pcd_in_image, pcd_in_cam, loc_yx_2d, width, height):

    # valid_mask = (pcd_in_image[:,0] >= 0) * (pcd_in_image[:,0] < width) * (pcd_in_image[:,1] >= 0) * (pcd_in_image[:,1] < height) * (pcd_in_image[:,2] > 0)

    pixel_locs = np.concatenate(
        [pcd_in_image[:, 1][:, None], pcd_in_image[:, 0][:, None]], 1)
    pixel_locs = pixel_locs.astype(int)

    eu_dist = np.sqrt(np.sum((pixel_locs - loc_yx_2d)**2, axis=1))
    closest_yx_idx = np.argmin(eu_dist)

    closest_pos_3D = pcd_in_cam[closest_yx_idx]
    closest_pos_2D = pixel_locs[closest_yx_idx]
    return closest_pos_3D, closest_pos_2D


def extract_rgb_from_image(
    pcd_in_image, pcd_in_cam, obj_masks, obj_dict, width, height
):

    valid_mask = \
        (0 <= pcd_in_image[:, 0]) * (pcd_in_image[:, 0] < width) * \
        (0 <= pcd_in_image[:, 1]) * (pcd_in_image[:, 1] < height) * \
        (0 < pcd_in_image[:, 2])

    pixel_locs = np.concatenate(
        [pcd_in_image[valid_mask, 1][:, None],
         pcd_in_image[valid_mask, 0][:, None]],
        axis=1)  # yx
    pixel_locs = pixel_locs.astype(int)
    valid_locs = np.where(valid_mask)[0]

    pcd_in_image_valid = pcd_in_image[valid_locs]

    colors = np.zeros((len(pcd_in_image_valid), 3))

    obj_mask_from_color = np.zeros((len(pcd_in_image_valid)))
    colors[:, :] = img[pixel_locs[:, 0], pixel_locs[:, 1]] / 255.0

    obj_points = []
    for obj_idx in range(len(obj_masks)):
        _, obj_id = obj_dict[obj_idx]
        obj_mask = obj_masks[obj_idx, 0]
        valid_locs_mask = np.where(
            obj_mask[pixel_locs[:, 0], pixel_locs[:, 1]])

        color_RGB = COLORS[colors_indice[obj_id]]['color']

        colors[valid_locs_mask, 0] = color_RGB[0] / 255.0
        colors[valid_locs_mask, 1] = color_RGB[1] / 255.0
        colors[valid_locs_mask, 2] = color_RGB[2] / 255.0
        obj_mask_from_color[valid_locs_mask] = obj_id

        obj_points.append(pcd_in_cam[valid_locs_mask])

    return colors, valid_mask, obj_points, obj_mask_from_color


def extract_rgb_from_image_pure(pcd_in_img, width, height):
    valid_mask = \
        (0 <= pcd_in_img[:, 0]) * (pcd_in_img[:, 0] < width) * \
        (0 <= pcd_in_img[:, 1]) * (pcd_in_img[:, 1] < height) * \
        (0 < pcd_in_img[:, 2])

    pixel_locs = np.concatenate(
        [pcd_in_img[valid_mask, 1][:, None],
            pcd_in_img[valid_mask, 0][:, None]],
        axis=1)  # yx
    pixel_locs = pixel_locs.astype(int)
    valid_locs = np.where(valid_mask)[0]

    pcd_colors = np.zeros((len(pcd_in_img), 3))
    pcd_colors[valid_mask, :] = img[pixel_locs[:, 0], pixel_locs[:, 1]] / 255.0

    return pcd_colors, valid_mask


def visualize_open3d(vis, pcd_with_rgb, obj_pos_colors,
                     obj_dict, cam_plotter=None, obj_points=[]):

    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(pcd_with_rgb[:, :3])
    all_pcd.colors = o3d.utility.Vector3dVector(pcd_with_rgb[:, 3:])
    vis.clear_geometries()

    if cam_plotter is not None:
        vis.add_geometry(cam_plotter)
    vis.add_geometry(all_pcd)

    if len(obj_points) > 0:

        meshes = []
        for mesh_idx in range(len(obj_points)):
            _, obj_id = obj_dict[mesh_idx]
            obj_pos = obj_points[mesh_idx]
            color_RGB = COLORS[colors_indice[obj_id]]['color']
            for p_idx in range(len(obj_pos)):

                mesh_on_surface = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.01)
                mesh_on_surface.paint_uniform_color(
                    [color_RGB[0] / 255.0, color_RGB[1] / 255.0, color_RGB[2] / 255.0])
                q_mat = Quaternion(0, 0, 0, 1)
                q_mat = q_mat.transformation_matrix

                # x:-backward/ +forward, z : -left/ +right
                q_mat[:3, -1] = np.array([obj_pos[p_idx, 0],
                                         obj_pos[p_idx, 1], obj_pos[p_idx, 2]])
                mesh_on_surface.transform(q_mat)
                vis.add_geometry(mesh_on_surface)
            # pdb.set_trace()
            # meshes.append(mesh_on_surface)
    viewpoint_param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
    vis.get_view_control().convert_from_pinhole_camera_parameters(viewpoint_param)

    # vis.run()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("tmp.png")

    img = cv2.imread('tmp.png')
    width, height = img.shape[1], img.shape[0]

    cut_length_left = int(width / 5)
    cut_length_right = int(width / 5)
    cut_length_top = int(height / 4)
    cut_length_bottom = int(height / 9)
    img = img[cut_length_top: -cut_length_bottom,
              cut_length_left: -cut_length_right]

    return img


def visualize_opencv(img, pcd_in_image, obj_masks,
                     obj_dict, width, height, use_mask=True):

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    valid_mask = (pcd_in_image[:,
                               0] >= 0) * (pcd_in_image[:,
                                                        0] < width) * (pcd_in_image[:,
                                                                                    1] >= 0) * (pcd_in_image[:,
                                                                                                             1] < height) * (pcd_in_image[:,
                                                                                                                                          2] > 0)

    pixel_locs = np.concatenate(
        [pcd_in_image[valid_mask, 1][:, None], pcd_in_image[valid_mask, 0][:, None]], 1)
    pixel_locs = pixel_locs.astype(int)

    if use_mask == False:
        img_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        img_mask[pixel_locs[:, 0], pixel_locs[:, 1]] = 1
        img_bgr *= img_mask
        img_bgr_canvas = np.copy(img_bgr)

        for valid_idx in range(len(pixel_locs)):

            color = img_bgr[pixel_locs[valid_idx, 0], pixel_locs[valid_idx, 1]]

            cv2.circle(img_bgr_canvas, (pixel_locs[valid_idx, 1], pixel_locs[valid_idx, 0]),
                       3, (int(color[0]), int(color[1]), int(color[2])), -1)
    else:
        img_bgr_canvas = np.zeros(
            (img.shape[0], img.shape[1], 3), dtype=np.uint8)

        '''
        for valid_idx in range(len(pixel_locs)):

            color = img_bgr[pixel_locs[valid_idx,0],pixel_locs[valid_idx,1]]

            cv2.circle(img_bgr_canvas, (pixel_locs[valid_idx,1], pixel_locs[valid_idx,0]),
                                                                3, (int(color[0]), int(color[1]), int(color[2])), -1)
        '''
        valid_locs = np.where(valid_mask)[0]
        for obj_idx in range(len(obj_masks)):
            _, obj_id = obj_dict[obj_idx]
            obj_mask = obj_masks[obj_idx, 0]
            valid_locs_mask = np.where(
                obj_mask[pixel_locs[:, 0], pixel_locs[:, 1]])

            valid_locs_img = pixel_locs[valid_locs_mask[0]]
            # valid_locs[valid_locs_mask]
            color_RGB = COLORS[colors_indice[obj_id]]['color']
            color_BGR = [color_RGB[2], color_RGB[1], color_RGB[0]]
            # img_bgr_canvas[valid_locs_img[:,0],valid_locs_img[:,1],:] = 255
            # pdb.set_trace()
            for valid_idx in range(len(valid_locs_img)):
                cv2.circle(img_bgr_canvas, (valid_locs_img[valid_idx, 1], valid_locs_img[valid_idx, 0]),
                           5, (color_BGR[0], color_BGR[1], color_BGR[2]), -1)
    return img_bgr_canvas


def sync_lidar_and_rgb(file_path_lidar, file_path_rgb):
    rgb_fpaths = sorted(glob.glob(os.path.join(file_path_rgb, '*.jpeg')))
    lidar_fpaths = sorted(glob.glob(os.path.join(file_path_lidar, '*.pcd')))

    rgb_list = []
    lidar_list = []

    for rgb_fpath in rgb_fpaths:
        rgb_filename = os.path.basename(rgb_fpath).split('.')[0]
        second_rgb, decimal_rgb = rgb_filename.split('_')[1:3]

        second_rgb = int(second_rgb)
        decimal_rgb = float('0.' + decimal_rgb)
        rgb_timestamp = second_rgb + decimal_rgb

        diff_list = []
        lidar_fp_list = []
        for lidar_fpath in lidar_fpaths:
            lidar_filename = os.path.basename(lidar_fpath).split('.')[0]
            _, _, second_lidar, decimal_lidar = lidar_filename.split('_')
            second_lidar = int(second_lidar)
            decimal_lidar = float('0.' + decimal_lidar)

            lidar_timestamp = second_lidar + decimal_lidar
            diff = abs(rgb_timestamp - lidar_timestamp)

            diff_list.append(diff)
            lidar_fp_list.append(lidar_fpath)

        diff_list = np.array(diff_list)
        matching_lidar_file = lidar_fp_list[np.argmin(diff_list)]

        rgb_list.append(rgb_fpath)
        assert os.path.exists(matching_lidar_file)
        lidar_list.append(matching_lidar_file)

    return lidar_list, rgb_list


def construct_cam_plotter(scale=1.5, translation=(0, 0, 10)):
    # Camera center is considered (0,0,0)
    # Here we construct four edges of camera. This is only used for
    # visualization.

    top_left = np.array([-scale + translation[0],
                         scale + translation[1],
                         0.0 + translation[2]])  # Top Left one
    top_right = np.array([scale +
                          translation[0], scale +
                          translation[1], -
                          0.0 +
                          translation[2]])  # Top Right one
    bottom_left = np.array([-scale + translation[0], -scale +
                           translation[1], 0.0 + translation[2]])  # Bottom Left one
    bottom_right = np.array([scale +
                             translation[0], -
                             scale +
                             translation[1], -
                             0.0 +
                             translation[2]])  # Bottom Right one
    focal_point = np.array(
        [0.0 + translation[0], 0.0 + translation[1], -scale * 2 + translation[2]])

    cam_plotter_array = np.array(
        [top_left, top_right, bottom_left, bottom_right, focal_point])

    return cam_plotter_array


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def get_2D_gt(gt_json_path):

    gt_json = json.load(open(gt_json_path, 'r'))
    img_dict = {}

    annotations = gt_json['annotations']
    images = gt_json['images']

    for idx in range(len(annotations)):

        anno = annotations[idx]
        bbox = anno['bbox']
        img_id = anno['image_id']
        obj_id = anno['category_id']
        img_filename = os.path.basename(images[img_id]['file_name'])

        if img_filename not in img_dict:
            img_dict[img_filename] = []
        img_dict[img_filename].append([bbox, obj_id])
    return img_dict


def main():
    # arguments
    data_dir = 'data/lion_sleep3'
    file_path_lidar = os.path.join(data_dir, 'lidar/')
    file_path_rgb = os.path.join(data_dir, 'sync_rgb/')
    file_path_mask = os.path.join(data_dir, 'masks_lion2/')
    cam_params = json.load(open(
        os.path.join(data_dir, 'manual_calibration.json'), 'r'))

    camera_plot_lines = [
        [4, 0], [4, 1], [4, 2], [4, 3],
        [0, 1], [1, 3], [3, 2], [2, 0]
    ]

    cam_plotter_pose = construct_cam_plotter()
    cam_plotter_colors = np.zeros((len(cam_plotter_pose), 3))

    cam_plotter = o3d.geometry.LineSet()
    cam_plotter_colors = np.zeros(cam_plotter_pose.shape)
    cam_plotter_colors[:, 0] = 1.0

    cam_plotter.points = o3d.utility.Vector3dVector(cam_plotter_pose)
    cam_plotter.lines = o3d.utility.Vector2iVector(camera_plot_lines)
    cam_plotter.colors = o3d.utility.Vector3dVector(cam_plotter_colors)

    lidar_list, rgb_list = sync_lidar_and_rgb(file_path_lidar, file_path_rgb)

    rot_mat = quaternion.as_rotation_matrix(quaternion.from_float_array(np.array([
        cam_params['extrinsics_R'][0],
        -cam_params['extrinsics_R'][1],
        -cam_params['extrinsics_R'][2],
        cam_params['extrinsics_R'][3]])))
    translation = np.array(cam_params['extrinsics_t'])

    fx, fy, cx, cy = cam_params['intrinsics']
    IMG_WIDTH, IMG_HEIGHT = 1280, 720

    OUTPUT_WIDTH, OUTPUT_HEIGHT = 3840, 720
    FPS = 4
    OUTPUT_VIDEO_PATH = os.path.join(data_dir, 'output.avi')
    output_dir = os.path.join(data_dir, 'textured_pcds/')

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    img_dict = get_2D_gt(os.path.join(data_dir, 'train.json'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS,
                          (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    for idx in tqdm(range(len(rgb_list))):
        # load the frame image
        rgb_path = rgb_list[idx]
        key = os.path.basename(rgb_path)
        img_bgr = cv2.imread(rgb_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # load the image mask
        mask_idx = np.load(file_path_mask + '{0}.npy'.format(idx))
        # load point cloud of the frame
        lidar_path = lidar_list[idx]
        pcd_in_lidar = o3d.io.read_point_cloud(lidar_path)
        pcd_in_lidar = np.asarray(pcd_in_lidar.points)

        # load the camera parameters
        intrinsic = make_intrinsic(fx, fy, cx, cy)
        extrinsic = make_extrinsic(rot_mat, translation)

        # project the point cloud to camera and its image sensor
        pcd_in_cam = lidar2cam_projection(pcd_in_lidar, extrinsic)
        pcd_in_img = cam2image_projection(pcd_in_cam, intrinsic)

        pcd_in_cam = pcd_in_cam.T[:, :-1]
        pcd_in_img = pcd_in_img.T[:, :-1]

        pcd_colors, valid_mask_save = extract_rgb_from_image_pure(
            pcd_in_img, width=IMG_WIDTH, height=IMG_HEIGHT)
        pcd_with_rgb_save = np.concatenate([pcd_in_cam, pcd_colors], 1)
        pcd_with_rgb_save = pcd_with_rgb_save[valid_mask_save]
        pcd_for_save_rgb = o3d.geometry.PointCloud()
        pcd_for_save_rgb.points = o3d.utility.Vector3dVector(
            pcd_with_rgb_save[:, :3])
        pcd_for_save_rgb.colors = o3d.utility.Vector3dVector(
            pcd_with_rgb_save[:, 3:])

        colors, valid_mask, obj_points_colors, obj_mask_from_color = extract_rgb_from_image(
            pcd_in_img, pcd_in_cam, mask_idx, img_dict[key], width=IMG_WIDTH, height=IMG_HEIGHT)

        # pcd_for_save_mask = o3d.geometry.PointCloud()
        # pcd_for_save_mask.points = o3d.utility.Vector3dVector(pcd_with_rgb_save[:,:3])
        file_prefix = rgb_path.split('/')[-1].split('.')[0]
        o3d.io.write_point_cloud(
            os.path.join(output_dir, file_prefix + '.pcd'),
            pcd_for_save_rgb)
        np.save(
            os.path.join(output_dir, file_prefix + '_mask.npy'),
            obj_mask_from_color)


if __name__ == '__main__':
    main()
