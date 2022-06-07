#!/usr/bin/python3

import carla
import open3d as o3d
from utils.geometry_types import *


def carla_location_to_numpy_vec(carla_location: carla.Location) -> numpy.array:
    """
    Convert a carla location to a ROS vector3

    Considers the conversion from left-handed system (unreal) to right-handed
    system

    :param carla_location: the carla location
    :type carla_location: carla.Location
    :return: a numpy.array (3x1 vector)
    :rtype: numpy.array
    """
    return numpy.array([[
        carla_location.x,
        -carla_location.y,
        carla_location.z
    ]]).reshape(3, 1)


def carla_location_to_location(carla_location: carla.Location) -> Location:
    """
    Convert a carla location to a ROS vector3

    Considers the conversion from left-handed system (unreal) to right-handed
    system

    :param carla_location: the carla location
    :type carla_location: carla.Location
    :return: a numpy.array (3x1 vector)
    :rtype: numpy.array
    """
    return Location(carla_location.x,
                    -carla_location.y,
                    carla_location.z)


def carla_rotation_to_RPY(carla_rotation):
    """
    Convert a carla rotation to a roll, pitch, yaw tuple

    Considers the conversion from left-handed system (unreal) to right-handed
    system.

    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a tuple with 3 elements (roll, pitch, yaw)
    :rtype: tuple
    """
    roll = carla_rotation.roll
    pitch = -carla_rotation.pitch
    yaw = -carla_rotation.yaw

    return (roll, pitch, yaw)


def carla_rotation_to_rotation(carla_rotation: carla.Rotation) -> Rotation:
    """
    Convert a carla rotation to rotation matrix.

    Considers the conversion from left-handed system (unreal) to right-handed
    system.
    Considers the conversion from degrees (carla) to radians.

    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a numpy.array with 3x3 elements
    :rtype: numpy.array
    """
    roll, pitch, yaw = carla_rotation_to_RPY(carla_rotation)
    return Rotation(roll=roll, pitch=pitch, yaw=yaw)


def carla_transform_to_transform(carla_transform: carla.Transform):
    """
    Convert a carla transform to transform type.
    Considers the conversion from left-handed system (unreal) to right-handed
    system.
    Considers the conversion from degrees (carla) to radians.

    :param carla_transform: the carla rotation
    :type carla_transform: carla.Rotation
    :return: a Transform type in right-hand axis
    :rtype: Transform
    """
    location = carla_location_to_location(carla_transform.location)
    rotation = carla_rotation_to_rotation(carla_transform.rotation)
    return Transform(location, rotation)


def carla_vec3d_to_numpy_vec(carla_vec3d: carla.Vector3D, left_to_right_hand=True):
    """
    Convert a carla vector3d to numpy 3x1 array.
    Considers the conversion from left-handed system (unreal) to right-handed
    system.

    :param carla_vec3d: the carla Vector3d
    :type carla_vec3d: carla.Vector3D
    :param left_to_right_hand: whether enable left-hand to right-hand convert
    :type left_to_right_hand: bool
    :return: a numpy.array with 3x1 elements
    :rtype: numpy.array
    """
    if left_to_right_hand:
        return numpy.array([[
            carla_vec3d.x,
            -carla_vec3d.y,
            carla_vec3d.z
        ]]).reshape(3, 1)
    else:
        return numpy.array([[
            carla_vec3d.x,
            carla_vec3d.y,
            carla_vec3d.z
        ]]).reshape(3, 1)


def carla_vec3d_to_vec3d(carla_vec3d: carla.Vector3D):
    return Vector3d(x=carla_vec3d.x, y=-carla_vec3d.y, z=carla_vec3d.z)


def RPY_to_carla_rotation(roll, pitch, yaw):
    return carla.Rotation(roll=math.degrees(roll),
                          pitch=-math.degrees(pitch),
                          yaw=-math.degrees(yaw))


def rotation_to_carla_rotation(rotation: Rotation):
    return carla.Rotation(roll=rotation.roll,
                          pitch=-rotation.pitch,
                          yaw=-rotation.yaw)


def location_to_carla_location(location: Location):
    return carla.Location(location.x, -location.y, location.z)


def transform_to_carla_transform(transform: Transform):
    carla_location = location_to_carla_location(transform.location)
    carla_rotation = rotation_to_carla_rotation(transform.rotation)
    return carla.Transform(carla_location, carla_rotation)


def carla_bbox_to_bbox(carla_bbox: carla.BoundingBox):
    location = carla_location_to_location(carla_bbox.location)
    rotation = carla_rotation_to_rotation(carla_bbox.rotation)
    extent = Vector3d(carla_bbox.extent.x,
                      carla_bbox.extent.y,
                      carla_bbox.extent.z)
    return BoundingBox(location, extent, rotation)


def bbox_to_o3d_bbox(bbox: BoundingBox):
    center = bbox.location.get_vector()
    rotation = bbox.rotation.get_rotation_matrix()
    extent = 2.0 * bbox.extent.get_vector()
    return o3d.geometry.OrientedBoundingBox(center, rotation, extent)
