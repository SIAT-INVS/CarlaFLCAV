#!/usr/bin/python3
import glob
import os.path
import sys

import cv2
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())
from utils.transform import *


def load_lidar_data(path: str):
    lidar_rawdata_path_list = sorted(glob.glob(f"{path}/*.npy"))
    lidar_rawdata_df = pd.DataFrame(columns=['frame', 'lidar_rawdata_path', 'lidar_pose'])
    lidar_poses = pd.read_csv(f"{path}/poses.csv")
    for lidar_rawdata_path in lidar_rawdata_path_list:
        frame = get_frame_from_fullpath(lidar_rawdata_path)
        row = lidar_poses[lidar_poses['frame'] == frame]
        lidar_pose = Transform(Location(float(row['x']), float(row['y']), float(row['z'])),
                               Rotation(yaw=float(row['yaw']),
                                        roll=float(row['roll']),
                                        pitch=float(row['pitch'])))
        lidar_rawdata_df = lidar_rawdata_df.append({'frame': frame,
                                                    'lidar_rawdata_path': lidar_rawdata_path,
                                                    'lidar_pose': lidar_pose},
                                                   ignore_index=True)
    return lidar_rawdata_df


def load_camera_data(path: str):
    camera_rawdata_list = sorted(glob.glob(f"{path}/*.png"))
    camera_poses = pd.read_csv(f"{path}/poses.csv")

    camera_info = pd.read_csv(f"{path}/camera_info.csv").loc[0]
    camera_matrix = np.array([[camera_info["fx"], 0.0, camera_info["cx"]],
                              [0.0, camera_info["fy"], camera_info["cy"]],
                              [0.0, 0.0, 1.0]])

    camera_rawdata_df = pd.DataFrame(columns=['frame', 'camera_rawdata_path', 'camera_pose', 'camera_matrix'])
    for camera_rawdata_path in camera_rawdata_list:
        frame = get_frame_from_fullpath(camera_rawdata_path)
        row = camera_poses[camera_poses['frame'] == frame]
        camera_pose = Transform(Location(float(row['x']), float(row['y']), float(row['z'])),
                                Rotation(yaw=float(row['yaw']),
                                         roll=float(row['roll']),
                                         pitch=float(row['pitch'])))
        camera_rawdata_df = camera_rawdata_df.append({'frame': frame,
                                                      'camera_pose': camera_pose,
                                                      'camera_matrix': camera_matrix,
                                                      'camera_rawdata_path': camera_rawdata_path},
                                                     ignore_index=True)
    return camera_rawdata_df


def get_frame_from_fullpath(path: str) -> int:
    return int(os.path.splitext(os.path.split(path)[-1])[0])


def load_object_labels(path: str):
    object_labels_path_list = sorted(glob.glob("{}/*.pkl".format(path)))
    object_labels_df = pd.DataFrame(columns=['frame', 'object_labels_path'])
    for objects_labels_rawdata_path in object_labels_path_list:
        frame = get_frame_from_fullpath(objects_labels_rawdata_path)
        object_labels_df = object_labels_df.append({'frame': frame,
                                                    'object_labels_path': objects_labels_rawdata_path},
                                                   ignore_index=True)
    return object_labels_df


def load_vehicle_pose(path: str) -> pd.DataFrame:
    vehicle_status_df = pd.read_csv("{}/vehicle_status.csv".format(path))
    vehicle_poses = []
    for idx, row in vehicle_status_df.iterrows():
        pose = Transform(Location(row['x'],
                                  row['y'],
                                  row['z']),
                         Rotation(roll=row['roll'],
                                  yaw=row['yaw'],
                                  pitch=row['pitch']))
        vehicle_poses.append(pose)
    vehicle_status_df["vehicle_pose"] = vehicle_poses
    return pd.DataFrame(vehicle_status_df, columns=['frame', 'timestamp', 'vehicle_pose'])


def read_pointcloud(path: str) -> np.array:
    pointcloud = np.load(path)
    return pointcloud


def read_image(path: str) -> np.array:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image
