#!/usr/bin/python3

import argparse
import glob
import os.path
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np
import open3d as o3d
from matplotlib import cm

sys.path.append(Path(__file__).parent.parent.as_posix())
from param import ROOT_PATH

VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


class PointcloudType(Enum):
        LIDAR = 0
        SEMANTIC_LIDAR = 1
        RADAR = 2


class LidarVisualizer:
    def __init__(self, pointcloud_type: PointcloudType, source: str):
        self.pointcloud_type = pointcloud_type
        self.source = source
        self.pcd = o3d.geometry.PointCloud()

    def visualize(self):
        if self.source.endswith('.npy'):
            raw_pcd = np.load(self.source)
            self.numpy_to_o3d(raw_pcd)
            o3d.visualization.draw_geometries([self.pcd])
        else:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Carla Lidar')
            # vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 1
            vis.get_render_option().show_coordinate_frame = True
            self.add_open3d_axis(vis)
            files = sorted(glob.glob("{}/*.npy".format(self.source)))
            frame = 0
            for file in files:
                # vis.clear_geometries()
                raw_pcd = np.load(file)
                self.numpy_to_o3d(raw_pcd)
                if frame == 0:
                    vis.add_geometry(self.pcd)
                vis.update_geometry(self.pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.1)
                frame += 1
            vis.destroy_window()

    def numpy_to_o3d(self, numpy_cloud):
        if self.pointcloud_type == PointcloudType.LIDAR:
            # Isolate the intensity and compute a color for it
            intensity = numpy_cloud[:, -1]
            intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
            int_color = np.c_[
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

            # Isolate the 3D data
            points = numpy_cloud[:, 0:3]
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(int_color)
            return True
        elif self.pointcloud_type == PointcloudType.SEMANTIC_LIDAR:
            # Read points
            points = np.array([numpy_cloud['x'], numpy_cloud['y'], numpy_cloud['z']]).T

            # Colorize the pointcloud based on the CityScapes color palette
            labels = np.array(numpy_cloud['ObjTag'])
            int_color = LABEL_COLORS[labels]

            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(int_color)
            return True
        elif self.pointcloud_type == PointcloudType.RADAR:
            self.pcd.points = o3d.utility.Vector3dVector(numpy_cloud[:, 0:3])
            return True
        else:
            return False

    def add_open3d_axis(self, vis):
        """Add a small 3D axis on Open3D Visualizer"""
        axis = o3d.geometry.LineSet()
        axis.points = o3d.utility.Vector3dVector(np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        axis.lines = o3d.utility.Vector2iVector(np.array([
            [0, 1],
            [0, 2],
            [0, 3]]))
        axis.colors = o3d.utility.Vector3dVector(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]))
        vis.add_geometry(axis)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--type',
        default='lidar',
        help='Type of point cloud (lidar / semantic_lidar / radar)'
    )
    argparser.add_argument(
        '--source',
        type=str,
        help='File or folder source to visualize'
    )

    args = argparser.parse_args()
    pointcloud_type = PointcloudType.LIDAR
    if args.type == 'lidar':
        pointcloud_type = PointcloudType.LIDAR
    elif args.type == 'semantic_lidar':
        pointcloud_type = PointcloudType.SEMANTIC_LIDAR
    elif args.type == 'radar':
        pointcloud_type = PointcloudType.RADAR
    else:
        print("Not valid point cloud type")
        raise RuntimeError

    source = args.source
    if not os.path.exists(source):
        source = "{}/{}".format(ROOT_PATH, source)
        if not os.path.exists(source):
            print("File or folder not exist: {}".format(source))
            raise RuntimeError
    print("Read data from: {}".format(source))

    lidar_visualizer = LidarVisualizer(pointcloud_type, source)
    lidar_visualizer.visualize()


if __name__ == "__main__":
    # execute only if run as a script
    main()
