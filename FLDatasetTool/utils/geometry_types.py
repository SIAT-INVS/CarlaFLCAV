#!/usr/bin/python3
import math
import numpy
import numpy as np
import transforms3d as tf3d
import open3d as o3d
import transforms3d.euler


class Vector3d(object):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def get_vector(self):
        return numpy.array([[
            self.x,
            self.y,
            self.z,
        ]], dtype=numpy.float32).reshape(3, 1)

    def to_dict(self, prefix='') -> dict:
        return {'{}x'.format(prefix): self.x,
                '{}y'.format(prefix): self.y,
                '{}z'.format(prefix): self.z}

    def to_str(self, name='Vector3d'):
        return "{}(x={}, y={}, z={})".format(name, self.x, self.y, self.z)

    def __eq__(self, other):
        return numpy.allclose(self.get_vector(),
                              other.get_vector())

    def __ne__(self, other):
        return not numpy.allclose(self.get_vector(),
                                  other.get_vector())

    def __str__(self):
        return "Vector3d(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Location(Vector3d):
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)

    def __str__(self):
        return "Location(x={}, y={}, z={})".format(self.x, self.y, self.z)


class Rotation:
    def __init__(self, *, pitch=0.0, yaw=0.0, roll=0.0, radian=False):
        # RPY in degree
        if radian:
            self.roll = math.degrees(float(roll))
            self.pitch = math.degrees(float(pitch))
            self.yaw = math.degrees(float(yaw))
        else:
            self.roll = roll
            self.pitch = pitch
            self.yaw = yaw

    def get_quaternion(self):
        quaternion = tf3d.euler.euler2quat(math.radians(self.roll),
                                           math.radians(self.pitch),
                                           math.radians(self.yaw))
        return quaternion

    def get_rotation_matrix(self):
        return tf3d.euler.euler2mat(math.radians(self.roll),
                                    math.radians(self.pitch),
                                    math.radians(self.yaw))

    def to_dict(self) -> dict:
        return {'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw}

    def __eq__(self, other):
        return numpy.allclose(self.get_rotation_matrix(),
                              other.get_rotation_matrix())

    def __ne__(self, other):
        return not numpy.allclose(self.get_rotation_matrix(),
                                  other.get_rotation_matrix())

    def __str__(self):
        return "Rotation(pitch={}, yaw={}, roll={})".format(self.pitch,
                                                            self.yaw,
                                                            self.roll)


class Transform:
    def __init__(self, location: Location, rotation: Rotation):
        self.location = location
        self.rotation = rotation

    def to_dict(self):
        location_dict = self.location.to_dict()
        rotation_dict = self.rotation.to_dict()
        location_dict.update(rotation_dict)
        return location_dict

    @staticmethod
    def create_transform_from_matrix(trans_mat: np.array):
        trans_vec = trans_mat[0:3, 3]
        rot_mat = trans_mat[0:3, 0:3]
        r, p, y = transforms3d.euler.mat2euler(rot_mat)
        location = Location(trans_vec[0], trans_vec[1], trans_vec[2])
        rotation = Rotation(roll=math.degrees(r), yaw=math.degrees(y), pitch=math.degrees(p))
        return Transform(location, rotation)

    @staticmethod
    def create_transform_from_Rt(r_mat: np.array, t_vec: np.array):
        r, p, y = transforms3d.euler.mat2euler(r_mat)
        location = Location(t_vec[0], t_vec[1], t_vec[2])
        rotation = Rotation(roll=math.degrees(r), yaw=math.degrees(y), pitch=math.degrees(p))
        return Transform(location, rotation)

    def get_matrix(self):
        t_vec = self.location.get_vector()
        r_mat = self.rotation.get_rotation_matrix()
        t_mat = numpy.concatenate((r_mat, t_vec), axis=1)
        t_mat_homo = numpy.concatenate((t_mat,
                                        numpy.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)
        return t_mat_homo

    def get_inverse_matrix(self):
        trans_mat = self.get_matrix()
        return numpy.linalg.inv(trans_mat)

    def get_forward_vector(self):
        f_vec = np.array([1.0, 0.0, 0.0, 1.0]).reshape(4, 1)
        f_vec_raw = np.matmul(self.get_matrix(), f_vec)
        norm = np.linalg.norm(f_vec_raw)
        if norm != 0:
            f_vec_raw /= norm
        return f_vec_raw[0:3]

    def get_up_vector(self):
        u_vec = np.array([0.0, 0.0, 1.0, 1.0]).reshape(4, 1)
        u_vec_raw = np.matmul(self.get_matrix(), u_vec)
        norm = np.linalg.norm(u_vec_raw)
        if norm != 0:
            u_vec_raw /= norm
        return u_vec_raw[0:3]

    def transform(self, point: Vector3d):
        trans_mat = self.get_matrix()
        p = numpy.concatenate((point.get_vector(), numpy.array([[1]])), axis=0)
        p = numpy.matmul(trans_mat, p, dtype=numpy.float)
        return Vector3d(p[0, 0], p[1, 0], p[2, 0])

    def inv_transform(self, point: Vector3d):
        trans_mat = self.get_inverse_matrix()
        p = numpy.concatenate((point.get_vector(), numpy.array([[1]])), axis=0)
        p = numpy.matmul(trans_mat, p, dtype=numpy.float)
        return Vector3d(p[0, 0], p[1, 0], p[2, 0])

    def __eq__(self, other):
        return (self.location == other.location) \
               and (self.rotation == other.rotation)

    def __ne__(self, other):
        return (self.location != other.location) \
               or (self.rotation != other.rotation)

    def __str__(self):
        return "Transform({}, {})".format(self.location, self.rotation)


class BoundingBox:
    def __init__(self, location: Location, extent: Vector3d, rotation=Rotation()):
        self.location = location
        self.extent = extent
        self.rotation = rotation

    def to_open3d(self):
        center = self.location.get_vector()
        rotation = self.rotation.get_rotation_matrix()
        extent = self.extent.get_vector()
        return o3d.geometry.OrientedBoundingBox(center, rotation, extent)

    def __str__(self):
        return "BoundingBox({}, {})".format(self.location, self.extent.to_str(name="Extent"), self.rotation)
