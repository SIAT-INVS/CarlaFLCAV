# #!/usr/bin/python3
# import glob
# import os.path
# import pickle
# import sys
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# sys.path.append(Path(__file__).parent.parent.as_posix())
# from param import RAW_DATA_PATH
# from utils.transform import *
#
#
#
#
#
# def gather_rawdata_to_dataframe(record_name: str, vehicle_path: str, lidar_path: str, camera_path: str):
#     rawdata_frames_df = pd.DataFrame()
#     vehicle_poses_df = load_vehicle_pose("{}/{}/{}".format(RAW_DATA_PATH, record_name, vehicle_path))
#     rawdata_frames_df = vehicle_poses_df
#
#     object_labels_path_df = load_object_labels("{}/{}/others.world_0".format(RAW_DATA_PATH, record_name))
#     rawdata_frames_df = pd.merge(rawdata_frames_df, object_labels_path_df, how='outer', on='frame')
#
#     lidar_rawdata_df = load_lidar_rawdata(f"{RAW_DATA_PATH}/{record_name}/{vehicle_path}/{lidar_path}")
#     rawdata_frames_df = pd.merge(rawdata_frames_df, lidar_rawdata_df, how='outer', on='frame')
#
#     camera_rawdata_path_df = load_camera_data(f"{RAW_DATA_PATH}/{record_name}/{vehicle_path}/{camera_path}")
#     rawdata_frames_df = pd.merge(rawdata_frames_df, camera_rawdata_path_df, how='outer', on='frame')
#
#     rawdata_frames_df.to_csv('/tmp/test.csv')
#     return rawdata_frames_df
#
# #
# # def main():
#     # rawdata_frames_df = gather_rawdata_to_dataframe("record_2022_0111_1624",
#     #                                                 "vehicle.tesla.model3_1",
#     #                                                 "sensor.lidar.ray_cast_4",
#     #                                                 "sensor.camera.rgb_2")
#
#     # vehicle_df = pd.DataFrame()
#     # for vehicle_path in vehicle_path_list:
#     #     camera_path = glob.glob("{}/sensor.camera.rgb_*".format(vehicle_path))[0]
#     #     lidar_path = glob.glob("{}/sensor.lidar.ray_cast_[0-999]".format(vehicle_path))[0]
#     #     vehicle_df = vehicle_df.append({"vehicle_path": vehicle_path,
#     #                                     "camera_path": camera_path,
#     #                                     "lidar_path": lidar_path}, ignore_index=True)
#     #
#     # for index, row in vehicle_df.iterrows():
#     #     vehicle_path = row['vehicle_path']
#     #     lidar_path = row['lidar_path']
#     #
#     #     lidar_poses_df = load_poses_csv("{}/poses.csv".format(lidar_path))
#     #     world_data_list = load_world_data("{}/{}/others.world_0".format(RAW_DATA_PATH, record_name))
#     #
#     #     lidar_poses_df['lidar_rawdata_path'] = load_lidar_data(lidar_path)
#     #     lidar_poses_df['world_data_path'] = world_data_list
#     #     print(lidar_poses_df)
#     #     # vehicle_df.insert(-1, 'world_data_path', world_data_list)
#     #
#     #     vis = o3d.visualization.Visualizer()
#     #     vis.create_window(window_name='Kitti Objects Label')
#     #     vis.get_render_option().point_size = 1
#     #     vis.get_render_option().show_coordinate_frame = True
#     #
#     #     frame = 0
#     #     # thread = ThreadPool()
#     #     # thread.starmap()
#     #     for i, frame in lidar_poses_df.iterrows():
#     #         vis.clear_geometries()
#     #         lidar_trans = Transform(Location(frame['x'],
#     #                                          frame['y'],
#     #                                          frame['z']),
#     #                                 Rotation(roll=frame['roll'],
#     #                                          yaw=frame['yaw'],
#     #                                          pitch=frame['pitch']))
#     #         lidar_data = numpy.load(frame['lidar_rawdata_path'])
#     #         print(frame['lidar_rawdata_path'])
#     #         o3d_pcd = o3d.geometry.PointCloud()
#     #         o3d_pcd.points = o3d.utility.Vector3dVector(lidar_data[:, 0:3])
#     #
#     #         # Load object labels from pickle data
#     #         with open(frame['world_data_path'], 'rb') as pkl_file:
#     #             objects_labels = pickle.load(pkl_file)
#     #
#     #         # Convert all bbox to o3d bbox
#     #         bbox_list = []
#     #         for label in objects_labels:
#     #             # Check Distance
#     #             dist = np.linalg.norm(lidar_trans.location.get_vector()-label.transform.location.get_vector())
#     #             if dist > 150 or dist < 5:
#     #                 continue
#     #
#     #             # Transform label bbox to lidar coordinate
#     #             world_to_lidar = lidar_trans.get_inverse_matrix()
#     #             label_to_world = label.transform.get_matrix()
#     #             label_in_lidar = np.matmul(world_to_lidar, label_to_world)
#     #             t_vec = label_in_lidar[0:3, -1]
#     #             r_mat = label_in_lidar[0:3, 0:3]
#     #             o3d_bbox = bbox_to_o3d_bbox(label.bounding_box)
#     #             o3d_bbox.rotate(r_mat)
#     #             o3d_bbox.translate(t_vec)
#     #             o3d_bbox.color = np.array([1.0, 0, 0])
#     #
#     #             # Check points in bbox
#     #             p_num_in_bbox = o3d_bbox.get_point_indices_within_bounding_box(o3d_pcd.points)
#     #             if len(p_num_in_bbox) < 10:
#     #                 continue
#     #
#     #             bbox_list.append(o3d_bbox)
#     #             vis.add_geometry(o3d_bbox)
#     #
#     #         vis.add_geometry(o3d_pcd)
#     #         vis.poll_events()
#     #         vis.update_renderer()
#
#
# if __name__ == '__main__':
#     main()
