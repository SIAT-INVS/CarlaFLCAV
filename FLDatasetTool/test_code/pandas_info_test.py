#!/usr/bin/python3
import carla
import pandas as pd
from utils.geometry_types import Transform, Location, Rotation


def main():
    df = pd.DataFrame(columns=['frame_id', 'timestamp', 'lidar_pose'])
    # frame_id, timestamp, lidar_pose, camera_pose
    dic = {
        'frame_id': 1,
        'timestamp': 120.0,
        'lidar_pose': Transform(Location(0, 0, 1), Rotation(roll=0.0, yaw=90.0, pitch=0.0))
    }
    df = df.append(dic, ignore_index=True)

    dic = {
        'frame_id': 2,
        'timestamp': 120.0,
        'lidar_pose': Transform(Location(0, 0, 1), Rotation(roll=0.0, yaw=90.0, pitch=0.0))
    }

    df = df.append(dic, ignore_index=True)
    print(df)

    df.to_pickle('test.pkl')

    new_df = pd.read_pickle('test.pkl')

    for idx, row in new_df.iterrows():
        print(row['lidar_pose'].get_matrix())


if __name__ == "__main__":
    main()
