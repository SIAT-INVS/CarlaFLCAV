#!/usr/bin/python3
import carla
from utils.transform import *


class PseudoActor(object):
    def __init__(self, uid, name, parent):
        if name == '':
            name = f"{self.get_type_id()}_{uid}"
        self.uid = uid
        self.name = name
        self.parent = parent

    def destroy(self):
        return True

    def get_type_id(self):
        raise NotImplementedError

    def get_carla_actor(self):
        return None

    def get_save_dir(self):
        raise NotImplementedError

    def get_uid(self):
        return self.uid

    def save_to_disk(self, frame_id, timestamp, debug=False):
        return

    def get_carla_transform(self) -> carla.Transform:
        raise NotImplementedError


class Actor(PseudoActor):
    def __init__(self, uid, name, parent, carla_actor: carla.Actor):
        self.carla_actor = carla_actor
        super(Actor, self).__init__(uid=uid,
                                    name=name,
                                    parent=parent)

    def destroy(self):
        # print("Destroying: uid={} name={} carla_id={}".format(self.uid, self.name, self.carla_actor.id))
        if self.carla_actor is not None:
            try:
                status = self.carla_actor.destroy()
                # time.sleep(1)
                # if status:
                #     print("-> success")
                return status
            except RuntimeError:
                # print("-> failed")
                return False

    def get_transform(self) -> Transform:
        trans = self.carla_actor.get_transform()
        return carla_transform_to_transform(trans)

    def set_transform(self, transform: Transform):
        trans = transform_to_carla_transform(transform)
        self.carla_actor.set_transform(trans)

    def get_acceleration(self) -> Vector3d:
        acc_world = carla_vec3d_to_vec3d(self.carla_actor.get_acceleration())
        trans = self.get_transform()
        trans.location = Location(0, 0, 0)
        acc_vehicle = trans.inv_transform(acc_world)
        return acc_vehicle

    def get_velocity(self) -> Vector3d:
        """
        Get actor velocity in Vector3D(vx, vy, vz). It it in Right-Hand coordinate.
        :return: Vector3D(vx, vy, vz)
        """
        vel_world = carla_vec3d_to_vec3d(self.carla_actor.get_velocity())
        trans = self.get_transform()
        trans.location = Location(0, 0, 0)
        vel_vehicle = trans.inv_transform(vel_world)
        return vel_vehicle

    def get_speed(self):
        """
        Get vehicle speed in km/h
        :return: vehicle speed in km/h
        """
        v = self.carla_actor.get_velocity()
        return 3.6 * math.sqrt(v.x * v.x
                               + v.y * v.y
                               + v.z * v.z)

    def get_type_id(self):
        return self.carla_actor.type_id

    def get_actor_id(self):
        return self.carla_actor.id

    def get_carla_actor(self):
        return self.carla_actor

    def get_save_dir(self):
        raise NotImplementedError

    def get_carla_transform(self):
        return self.carla_actor.get_transform()



