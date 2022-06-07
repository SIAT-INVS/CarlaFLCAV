#!/usr/bin/python3
import copy
import numpy as np
import open3d as o3d
import cv2
import os
import sys
from pathlib import Path
import transforms3d.euler

sys.path.append(Path(__file__).parent.parent.as_posix())
from utils.geometry_types import Transform, Location
from utils.label_types import ObjectLabel
from utils.transform import bbox_to_o3d_bbox


class Param:
    POINTS_MIN = 20
    RANGE_MIN = 1.0
    RANGE_MAX = 150.0


def transform_lidar_point_to_cam(point, lidar_trans: Transform, cam_trans: Transform):
    p = np.append(point, [1.0])
    T_wl = lidar_trans.get_matrix()
    T_cw = cam_trans.get_inverse_matrix()
    T_cl = np.matmul(T_cw, T_wl)
    p_c = np.matmul(T_cl, p)
    return p_c


def project_point_to_image(point_in_cam,
                           cam_mat: np.array):
    p_c = point_in_cam
    p_c = p_c[0:3] / p_c[2]
    p_uv = np.matmul(cam_mat, p_c)
    p_uv = p_uv[0:2].astype(int)
    return p_uv


def transform_o3d_bbox(o3d_bbox: o3d.geometry.OrientedBoundingBox, transform_mat: np.array):
    o3d_bbox.rotate(transform_mat[0:3, 0:3], np.array([0, 0, 0]))
    o3d_bbox.translate(transform_mat[0:3, 3])
    return o3d_bbox


def bbox_to_o3d_bbox_in_target_coordinate(label: ObjectLabel, target_transform: Transform):
    world_to_target = target_transform.get_inverse_matrix()
    label_to_world = label.transform.get_matrix()
    label_in_target = np.matmul(world_to_target, label_to_world)
    o3d_bbox = bbox_to_o3d_bbox(label.bounding_box)
    o3d_bbox = transform_o3d_bbox(o3d_bbox, label_in_target)
    o3d_bbox.color = np.array([1.0, 0, 0])
    return o3d_bbox


def o3d_bbox_rotation_to_rpy(o3d_bbox: o3d.geometry.OrientedBoundingBox):
    # o3d bbox R to euler RPY in radian
    roll, pitch, yaw = transforms3d.euler.mat2euler(o3d_bbox.R)
    return roll, pitch, yaw


def cal_truncated(image_length, image_width, bbox_2d: list) -> float:
    # Calculate truncated.
    # from 0 (non-truncated) to 1 (truncated), where
    # truncated refers to the object leaving image boundaries

    # [x_min y_min x_max y_max]
    bbox_2d_in_img = copy.deepcopy(bbox_2d)
    bbox_2d_in_img[0] = max(bbox_2d[0], 0)
    bbox_2d_in_img[1] = max(bbox_2d[1], 0)
    bbox_2d_in_img[2] = min(bbox_2d[2], image_width)
    bbox_2d_in_img[3] = min(bbox_2d[3], image_length)

    size1 = (bbox_2d_in_img[2] - bbox_2d_in_img[0]) * (bbox_2d_in_img[3] - bbox_2d_in_img[1])
    size2 = (bbox_2d[2] - bbox_2d[0]) * (bbox_2d[3] - bbox_2d[1])
    truncated = size1 / size2
    truncated = max(truncated, 0.0)
    truncated = min(truncated, 1.0)
    truncated = 1.0 - truncated
    return truncated


def cal_occlusion(pcd: o3d.geometry.PointCloud, bbox_3d: o3d.geometry.OrientedBoundingBox):
    occlusion = 2
    p_in_bbox = bbox_3d.get_point_indices_within_bounding_box(pcd.points)
    p_num = len(p_in_bbox)
    if p_num < Param.POINTS_MIN:
        occlusion = -1
        return occlusion
    elif p_num > Param.POINTS_MIN:
        occlusion = 0
    if p_num + bbox_3d.center[0] < 250:
        occlusion = 1
    if p_num + bbox_3d.center[0] < 125:
        occlusion = 2
    return occlusion


def is_valid_distance(source_location: Location, target_location: Location):
    dist = np.linalg.norm(source_location.get_vector() - target_location.get_vector())
    if Param.RANGE_MIN < dist < Param.RANGE_MAX:
        return True
    else:
        return False


def write_pointcloud(output_dir: str, frame_id: str, lidar_data: np.array):
    lidar_dir = f"{output_dir}/velodyne"
    os.makedirs(lidar_dir, exist_ok=True)
    file_path = "{}/{}.bin".format(lidar_dir, frame_id)
    lidar_data.tofile(file_path)


def write_image(output_dir: str, frame_id: str, image: np.array):
    image_dir = f"{output_dir}/image_2"
    os.makedirs(image_dir, exist_ok=True)
    file_path = "{}/{}.png".format(image_dir, frame_id)
    cv2.imwrite(file_path, image)


def write_label(output_dir, frame_id, kitti_labels):
    label_dir = f"{output_dir}/label_2"
    os.makedirs(label_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(label_dir, frame_id)

    if len(kitti_labels) < 1:
        kitti_labels.append('DontCare -1 -1 -10 522.25 202.35 547.77 219.71 -1 -1 -1 -1000 -1000 -1000 -10 -10 \n')

    with open(file_path, 'w') as label_file:
        label_file.writelines(kitti_labels)


def write_calib(output_dir, frame_id, lidar_trans: Transform, cam_trans: Transform, camera_mat: np.array):
    """ Saves the calibration matrices to a file.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame.
    """
    calib_dir = f"{output_dir}/calib"
    os.makedirs(calib_dir, exist_ok=True)
    file_path = "{}/{}.txt".format(calib_dir, frame_id)

    camera_mat = np.concatenate((camera_mat, np.array([[0.0], [0.0], [0.0]])), axis=1)
    camera_mat = camera_mat.reshape(1, 12)
    camera_mat_str = ""
    for x in camera_mat[0]:
        camera_mat_str += str(x)
        camera_mat_str += ' '
    camera_mat_str += '\n'

    calib_str = list()
    calib_str.append(f"P0: {camera_mat_str}")
    calib_str.append(f"P1: {camera_mat_str}")
    calib_str.append(f"P2: {camera_mat_str}")
    calib_str.append(f"P3: {camera_mat_str}")

    calib_str.append("R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 \n")

    velo_to_cam = np.matmul(cam_trans.get_inverse_matrix(), lidar_trans.get_matrix())
    velo_to_cam = velo_to_cam[0:3, :]
    velo_to_cam = velo_to_cam.reshape(1, 12).tolist()
    velo_to_cam_str = "Tr_velo_to_cam: "
    for x in velo_to_cam[0]:
        velo_to_cam_str += str(x)
        velo_to_cam_str += ' '
    velo_to_cam_str += '\n'

    calib_str.append(velo_to_cam_str)

    with open(file_path, 'w') as calib_file:
        calib_file.writelines(calib_str)
        calib_file.close()


def generate_kitti_labels(label_type: str,
                          truncated: float,
                          occlusion: float,
                          alpha: float,
                          bbox_2d: list,
                          bbox_3d: o3d.geometry.OrientedBoundingBox,
                          rotation_y: float):
    # Note: Kitti Object 3d bbox location is top-plane-center, not the bbox center
    label_str = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} \n".format(label_type, truncated, occlusion, alpha,
                                                                         bbox_2d[0], bbox_2d[1],
                                                                         bbox_2d[2], bbox_2d[3],
                                                                         bbox_3d.extent[2],
                                                                         bbox_3d.extent[1],
                                                                         bbox_3d.extent[0],
                                                                         bbox_3d.center[0],
                                                                         bbox_3d.center[1] + (bbox_3d.extent[2] / 2.0),
                                                                         bbox_3d.center[2],
                                                                         rotation_y)
    return label_str
