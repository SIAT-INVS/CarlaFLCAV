#!/usr/bin/python3
import argparse
import math
import pickle
import sys
import time
from multiprocessing.pool import Pool as ThreadPool
from pathlib import Path

import numpy as np

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import RAW_DATA_PATH, DATASET_PATH
from label_tools.kitti_object.kitti_object_data_loader import *
from label_tools.kitti_object.kitti_object_helper import *


def gather_rawdata_to_dataframe(record_name: str, vehicle_name: str, lidar_path: str, camera_path: str):
    rawdata_frames_df = pd.DataFrame()
    # vehicle_poses_df = load_vehicle_pose("{}/{}/{}".format(RAW_DATA_PATH, record_name, vehicle_name))
    # rawdata_frames_df = vehicle_poses_df

    object_labels_path_df = load_object_labels("{}/{}/others.world_0".format(RAW_DATA_PATH, record_name))
    rawdata_frames_df = object_labels_path_df
    rawdata_frames_df = rawdata_frames_df.reset_index(drop=False)

    lidar_rawdata_df = load_lidar_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}/{lidar_path}")
    rawdata_frames_df = pd.merge(rawdata_frames_df, lidar_rawdata_df, how='outer', on='frame')

    camera_rawdata_path_df = load_camera_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_name}/{camera_path}")
    rawdata_frames_df = pd.merge(rawdata_frames_df, camera_rawdata_path_df, how='outer', on='frame')

    return rawdata_frames_df


def generate_image_sets(path_to_kitti_object: str):
    calib_data_list = sorted(glob.glob(f"{path_to_kitti_object}/training/calib/*.txt"))
    image_sets = []
    for item in calib_data_list:
        frame_id = os.path.splitext(os.path.split(item)[-1])[0]
        image_sets.append(f"{frame_id}\n")
    image_sets_dir = f"{path_to_kitti_object}/ImageSets"
    os.makedirs(image_sets_dir, exist_ok=True)
    with open(f"{image_sets_dir}/train.txt", 'w') as file:
        file.writelines(image_sets)
    with open(f"{image_sets_dir}/val.txt", 'w') as file:
        file.writelines(image_sets)


class KittiObjectLabelTool:
    def __init__(self, record_name, vehicle_name, rawdata_df: pd.DataFrame, output_dir=None):
        self.record_name = record_name
        self.vehicle_name = vehicle_name
        self.rawdata_df = rawdata_df
        self.range_max = 150.0
        self.range_min = 1.0
        self.points_min = 10
        self.output_dir = output_dir

    def process(self):
        debug = False
        if not debug:
            start = time.time()
            thread_pool = ThreadPool()
            thread_pool.starmap(self.process_frame, self.rawdata_df.iterrows())
            thread_pool.close()
            thread_pool.join()

            if self.output_dir is '':
                output_dir = f"{DATASET_PATH}/{self.record_name}/{self.vehicle_name}/kitti_object"
            else:
                output_dir = f"{DATASET_PATH}/{self.output_dir}/kitti_object"
            generate_image_sets(output_dir)
            print("Cost: {:0<3f}s".format(time.time() - start))
        else:
            start = time.time()
            for index, frame in self.rawdata_df.iterrows():
                self.process_frame(index, frame)
            if self.output_dir is '':
                output_dir = f"{DATASET_PATH}/{self.record_name}/{self.vehicle_name}/kitti_object"
            else:
                output_dir = f"{DATASET_PATH}/{self.output_dir}/kitti_object"
            generate_image_sets(output_dir)
            print("Cost: {:0<3f}s".format(time.time()-start))

    def process_frame(self, index, frame):
        index = "{:0>6d}".format(index)
        frame_id = "{:0>6d}".format(frame['frame'])
        lidar_trans: Transform = frame['lidar_pose']
        cam_trans: Transform = frame['camera_pose']
        cam_mat = np.asarray(frame['camera_matrix'])

        image = read_image(frame['camera_rawdata_path'])

        pointcloud_raw = read_pointcloud(frame['lidar_rawdata_path'])
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pointcloud_raw[:, 0:3])

        # Load object labels from pickle data
        with open(frame['object_labels_path'], 'rb') as pkl_file:
            objects_labels = pickle.load(pkl_file)

        # Transform matrix to transform object from lidar to camera coordinate
        T_lc = np.matmul(cam_trans.get_inverse_matrix(), lidar_trans.get_matrix())

        bbox_list_3d = []
        bbox_list_2d = []
        kitti_labels = []
        for label in objects_labels:
            # Kitti Object - Type
            if label.label_type == 'vehicle':
                label_type = 'Car'
            elif label.label_type:
                label_type = 'Pedestrian'
            else:
                label_type = 'DontCare'

            if not is_valid_distance(lidar_trans.location, label.transform.location):
                continue

            # Convert object label to open3d bbox type in lidar coordinate
            o3d_bbox = bbox_to_o3d_bbox_in_target_coordinate(label, lidar_trans)

            # Check lidar points in bbox
            occlusion = cal_occlusion(o3d_pcd, o3d_bbox)
            if occlusion < 0:
                continue

            # Transform bbox vertices to camera coordinate
            vertex_points = np.asarray(o3d_bbox.get_box_points())
            bbox_points_2d_x = []
            bbox_points_2d_y = []
            # bbox_points_3d = []
            for p in vertex_points:
                p_c = transform_lidar_point_to_cam(p, lidar_trans, cam_trans)
                # bbox_points_3d.append(p_c)
                p_uv = project_point_to_image(p_c, cam_mat)
                bbox_points_2d_x.append(p_uv[0])
                bbox_points_2d_y.append(p_uv[1])

            x_min = min(bbox_points_2d_x)
            x_max = max(bbox_points_2d_x)
            y_min = min(bbox_points_2d_y)
            y_max = max(bbox_points_2d_y)
            bbox_2d = [x_min, y_min, x_max, y_max]
            # For Debug
            # Draw 2d bbox
            # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=1)

            truncated = cal_truncated(image.shape[0], image.shape[1], bbox_2d)

            # Ignore backward vehicles
            if o3d_bbox.center[0] < 0:
                truncated = 1.0

            o3d_bbox = bbox_to_o3d_bbox_in_target_coordinate(label, cam_trans)

            rotation_y = -math.radians(label.transform.rotation.yaw - cam_trans.rotation.yaw)
            rotation_y = math.atan2(math.sin(rotation_y), math.cos(rotation_y))

            bbox_center = np.asarray(o3d_bbox.center)
            theta = math.atan2(-bbox_center[0], bbox_center[2])
            alpha = rotation_y - theta
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))

            kitti_label = generate_kitti_labels(label_type, truncated, occlusion, alpha,
                                                bbox_2d, o3d_bbox, rotation_y)

            kitti_labels.append(kitti_label)
            bbox_list_3d.append(o3d_bbox)
            bbox_list_2d.append(bbox_2d)

        # Preview each frame label result
        # if index < 35:
        #     return
        # o3d_pcd.rotate(T_lc[0:3, 0:3], np.array([0, 0, 0]))
        # o3d_pcd.translate(T_lc[0:3, 3])
        # cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        # preview_obj = [o3d_pcd, cam_coord]
        # if len(bbox_list_3d) <= 0:
        #     return
        # for bbox3d in bbox_list_3d:
        #     center = np.asarray(bbox3d.center)
        #     box_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=center)
        #     box_coord.rotate(bbox3d.R)
        #     preview_obj.append(box_coord)
        #     preview_obj.append(bbox3d)
        # print(frame_id)
        # o3d.visualization.draw_geometries(preview_obj)

        # Output dataset in kitti format
        if self.output_dir is '':
            output_dir = f"{DATASET_PATH}/{self.record_name}/{self.vehicle_name}/kitti_object/training"
        else:
            output_dir = f"{DATASET_PATH}/{self.output_dir}/kitti_object/training"
        write_calib(output_dir, index, lidar_trans, cam_trans, cam_mat)
        write_label(output_dir, index, kitti_labels)
        write_image(output_dir, index, image)
        write_pointcloud(output_dir, index, pointcloud_raw)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--record', '-r',
        required=True,
        help='Rawdata Record ID. e.g. record_2022_0113_1337'
    )
    argparser.add_argument(
        '--vehicle', '-v',
        default='all',
        help='Vehicle name. e.g. `vehicle.tesla.model3_1`. Default to all vehicles. '
    )
    argparser.add_argument(
        '--lidar', '-l',
        default='velodyne',
        help='Lidar name. e.g. sensor.lidar.ray_cast_4'
    )
    argparser.add_argument(
        '--camera', '-c',
        default='image_2',
        help='Camera name. e.g. sensor.camera.rgb_2'
    )
    argparser.add_argument(
        '--output_dir', '-o',
        default='',
        help='Output dir in dataset folder'
    )

    args = argparser.parse_args()

    record_name = args.record
    if args.vehicle == 'all':
        vehicle_name_list = [os.path.basename(x) for x in glob.glob('{}/{}/vehicle.*'.format(RAW_DATA_PATH, record_name))]
    else:
        vehicle_name_list = [args.vehicle]

    for vehicle_name in vehicle_name_list:
        rawdata_df = gather_rawdata_to_dataframe(args.record,
                                                 vehicle_name,
                                                 args.lidar,
                                                 args.camera)
        print("Process {} - {}".format(record_name, vehicle_name))
        kitti_obj_label_tool = KittiObjectLabelTool(record_name, vehicle_name, rawdata_df, args.output_dir)
        kitti_obj_label_tool.process()


if __name__ == '__main__':
    main()
