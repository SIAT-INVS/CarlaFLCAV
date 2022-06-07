#!/usr/bin/python3
import sys
from pathlib import Path
import carla
import numpy

sys.path.append(Path(__file__).parent.parent.as_posix())
from utils.transform import *


def main():
    print("---------------------------")
    print("CarlaTypes (left-hand): ")
    carla_location = carla.Location(1, 2, 3)
    carla_rotation = carla.Rotation(30, 60, 90)
    carla_transform = carla.Transform(carla_location, carla_rotation)
    print("{}\n{}\n{}".format(carla_location, carla_rotation, carla_transform))
    print("---------------------------")
    print("CustomTypes (right-hand): ")
    location = carla_location_to_location(carla_location)
    rotation = carla_rotation_to_rotation(carla_rotation)
    transform = carla_transform_to_transform(carla_transform)
    print(location)
    print(rotation)
    print(transform)
    print("---------------------------")
    carla_location_1 = location_to_carla_location(location)
    carla_rotation_1 = rotation_to_carla_rotation(rotation)
    carla_transform_1 = transform_to_carla_transform(transform)
    print(carla_location_1 == carla_location and
          carla_rotation_1 == carla_rotation and
          carla_transform_1 == carla_transform)

    # Point in reference frame
    point = Location(3, 2, 1)
    # Reference frame in world coordinate
    ref_coord = Transform(Location(1, 2, 3), Rotation(pitch=30.0, roll=60.0, yaw=90.0))
    # Transform point to world coordinate
    trans_point = ref_coord.transform(point)

    point_c = location_to_carla_location(point)
    ref_coord_c = transform_to_carla_transform(ref_coord)
    trans_point_c = ref_coord_c.transform(point_c)

    p1 = trans_point
    p2 = carla_location_to_location(trans_point_c)
    print(p1 == p2)


if __name__ == "__main__":
    main()
