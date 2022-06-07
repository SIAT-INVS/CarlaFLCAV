import numpy as np
import struct
import open3d as o3d
import argparse

def convert_kitti_bin_to_pcd(filepath: str):
    size_float = 4
    list_pcd = []
    with open(filepath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-s',
        type=str,
        help='File to visualize'
    )

    args = argparser.parse_args()
    pcd = convert_kitti_bin_to_pcd(args.s)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
