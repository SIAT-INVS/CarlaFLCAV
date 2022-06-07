#!/usr/bin/python3
import os
import json
import random
import warnings
from enum import Enum

import carla

from param import RAW_DATA_PATH, ROOT_PATH
from utils.geometry_types import *
from utils.transform import transform_to_carla_transform

from recorder.actor import Actor, PseudoActor
from recorder.camera import RgbCamera, DepthCamera, SemanticSegmentationCamera
from recorder.lidar import Lidar, SemanticLidar
from recorder.radar import Radar
from recorder.vehicle import Vehicle, OtherVehicle
from recorder.infrastructure import Infrastructure
from recorder.world import WorldActor


class NodeType(Enum):
    DEFAULT = 0
    WORLD = 1
    VEHICLE = 2
    INFRASTRUCTURE = 3
    SENSOR = 4
    OTHER_VEHICLE = 5


class Node(object):
    def __init__(self, actor=None, node_type=NodeType.DEFAULT):
        self._actor = actor
        self._node_type = node_type
        self._children_nodes = []

    def add_child(self, node):
        self._children_nodes.append(node)

    def get_actor(self):
        return self._actor

    def get_children(self):
        return self._children_nodes

    def get_node_type(self):
        return self._node_type

    def destroy(self):
        for node in self._children_nodes:
            node.destroy()
        if self._actor is not None:
            self._actor.destroy()

    # Tick for control step, running before world.tick()
    def tick_controller(self):
        if self._node_type == NodeType.VEHICLE or \
                self._node_type == NodeType.OTHER_VEHICLE:
            self._actor.control_step()

    def tick_data_saving(self, frame_id, timestamp):
        if self.get_node_type() == NodeType.SENSOR \
                or NodeType.VEHICLE \
                or NodeType.WORLD:
            self._actor.save_to_disk(frame_id, timestamp, True)


def get_name_from_json(json_info, name_set: set):
    # Get actor name from json, default to ''.
    # Actor will generate unique name "[TYPE_ID]_[UID]" later for saving data.
    try:
        name = str(json_info.pop("name"))
    except (KeyError, AttributeError):
        name = ''

    # If there is a same name in the name set, fallback to default
    if name != '':
        if name in name_set:
            warnings.warn(f"Invalid duplicated name {name}, fallback to default.")
            name = ''
        else:
            name_set.add(name)

    return name


def create_spawn_point(x, y, z, roll, pitch, yaw):
    return transform_to_carla_transform(Transform(Location(x, y, z), Rotation(roll=roll, pitch=pitch, yaw=yaw)))


class ActorFactory(object):
    def __init__(self, world: carla.World, base_save_dir=None):
        self._uid_count = 0
        self.world = world
        self.blueprint_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.base_save_dir = base_save_dir
        self.v2x_layer_name_set = set()
        self.sensor_layer_name_set = set()

    def create_actor_tree(self, actor_config_file):
        assert (self.base_save_dir is not None)
        if not actor_config_file or not os.path.exists(actor_config_file):
            raise RuntimeError(
                "Could not read actor config file from {}".format(actor_config_file))
        with open(actor_config_file) as handle:
            json_actors = json.loads(handle.read())

        root = self.create_world_node()
        for actor_info in json_actors["actors"]:
            actor_type = str(actor_info["type"])
            node = Node()
            if actor_type.startswith("vehicle"):
                node = self.create_vehicle_node(actor_info)
                root.add_child(node)
            elif actor_type.startswith("infrastructure"):
                node = self.create_infrastructure_node(actor_info)
                root.add_child(node)
            if node is not None:
                # If it has sensor setting, then create subtree
                if actor_info["sensors_setting"] is not None:
                    sensor_info_file = actor_info["sensors_setting"]
                    with open("{}/config/{}".format(ROOT_PATH, sensor_info_file)) as sensor_handle:
                        sensors_setting = json.loads(sensor_handle.read())
                        sensor_name_set = set()
                        for sensor_info in sensors_setting["sensors"]:
                            sensor_node = self.create_sensor_node(sensor_info, node.get_actor(), sensor_name_set)
                            node.add_child(sensor_node)

        other_vehicle_info = json_actors["other_vehicles"]
        ov_nodes = self.create_other_vehicles(other_vehicle_info)
        root.get_children().extend(ov_nodes)

        return root

    def create_world_node(self):
        world_actor = WorldActor(uid=self.generate_uid(),
                                 carla_world=self.world,
                                 base_save_dir=self.base_save_dir)
        world_node = Node(world_actor, NodeType.WORLD)
        return world_node

    def create_vehicle_node(self, actor_info):
        vehicle_type = actor_info["type"]
        vehicle_name = get_name_from_json(actor_info, self.v2x_layer_name_set)
        spawn_point = actor_info["spawn_point"]
        if type(spawn_point) is int:
            transform = self.spawn_points[spawn_point]
        else:
            transform = create_spawn_point(
                spawn_point.pop("x", 0.0),
                spawn_point.pop("y", 0.0),
                spawn_point.pop("z", 0.0),
                spawn_point.pop("roll", 0.0),
                spawn_point.pop("pitch", 0.0),
                spawn_point.pop("yaw", 0.0))
        blueprint = self.blueprint_lib.find(vehicle_type)
        carla_actor = self.world.spawn_actor(blueprint, transform)
        print(vehicle_name)
        vehicle_object = Vehicle(uid=self.generate_uid(),
                                 name=vehicle_name,
                                 base_save_dir=self.base_save_dir,
                                 carla_actor=carla_actor)
        vehicle_node = Node(vehicle_object, NodeType.VEHICLE)
        return vehicle_node

    def create_other_vehicles(self, other_vehicles_info):
        blueprints = self.blueprint_lib.filter('vehicle.*')
        other_vehicle_nodes = []
        try:
            spawn_points = other_vehicles_info['spawn_points']
        except (KeyError, ValueError):
            spawn_points = None

        if spawn_points:
            for spawn_point in spawn_points:
                bp = random.choice(blueprints)
                transform = self.spawn_points[spawn_point]
                carla_actor = self.world.spawn_actor(bp, transform)
                other_vehicle_object = OtherVehicle(uid=self.generate_uid(),
                                                    name='',
                                                    base_save_dir="/tmp",
                                                    carla_actor=carla_actor)
                other_vehicle_node = Node(other_vehicle_object, NodeType.OTHER_VEHICLE)
                other_vehicle_nodes.append(other_vehicle_node)

        try:
            vehicle_num = other_vehicles_info['vehicle_num']
        except (KeyError, ValueError):
            vehicle_num = None
        if vehicle_num:
            for i in range(vehicle_num):
                bp = random.choice(blueprints)
                all_spawn_points = self.world.get_map().get_spawn_points()
                try:
                    carla_actor = self.world.spawn_actor(bp, random.choice(all_spawn_points))
                except RuntimeError:
                    i -= 1
                    continue

                other_vehicle_object = OtherVehicle(uid=self.generate_uid(),
                                                    name='',
                                                    base_save_dir="/tmp",
                                                    carla_actor=carla_actor)
                other_vehicle_node = Node(other_vehicle_object, NodeType.OTHER_VEHICLE)
                other_vehicle_nodes.append(other_vehicle_node)

        return other_vehicle_nodes

    def create_infrastructure_node(self, actor_info):
        infrastructure_name = get_name_from_json(actor_info, self.v2x_layer_name_set)
        spawn_point = actor_info["spawn_point"]
        if type(spawn_point) is int:
            transform = self.spawn_points[spawn_point]
        else:
            transform = create_spawn_point(
                spawn_point.pop("x", 0.0),
                spawn_point.pop("y", 0.0),
                spawn_point.pop("z", 0.0),
                0,
                0,
                0,
            )
        infrastructure_object = Infrastructure(uid=self.generate_uid(),
                                               name=infrastructure_name,
                                               base_save_dir=self.base_save_dir,
                                               transform=transform)
        infrastructure_node = Node(infrastructure_object, NodeType.INFRASTRUCTURE)
        return infrastructure_node

    def create_sensor_node(self, sensor_info: dict, parent_actor: PseudoActor, sensor_name_set: set):
        sensor_type = str(sensor_info.pop("type"))
        sensor_name = get_name_from_json(sensor_info, sensor_name_set)
        spawn_point = sensor_info.pop("spawn_point")
        sensor_transform = create_spawn_point(
            spawn_point.pop("x", 0.0),
            spawn_point.pop("y", 0.0),
            spawn_point.pop("z", 0.0),
            spawn_point.pop("roll", 0.0),
            spawn_point.pop("pitch", 0.0),
            spawn_point.pop("yaw", 0.0))
        blueprint = self.blueprint_lib.find(sensor_type)
        for attribute, value in sensor_info.items():
            blueprint.set_attribute(attribute, str(value))
        if parent_actor.get_carla_actor() is not None:
            carla_actor = self.world.spawn_actor(blueprint, sensor_transform, parent_actor.get_carla_actor())
        else:
            sensor_location = parent_actor.get_carla_transform().transform(sensor_transform.location)
            sensor_transform = carla.Transform(sensor_location, sensor_transform.rotation)
            carla_actor = self.world.spawn_actor(blueprint, sensor_transform)

        sensor_actor = None
        if sensor_type == 'sensor.camera.rgb':
            sensor_actor = RgbCamera(uid=self.generate_uid(),
                                     name=sensor_name,
                                     base_save_dir=parent_actor.get_save_dir(),
                                     carla_actor=carla_actor,
                                     parent=parent_actor)
        elif sensor_type == 'sensor.camera.depth':
            sensor_actor = DepthCamera(uid=self.generate_uid(),
                                       name=sensor_name,
                                       base_save_dir=parent_actor.get_save_dir(),
                                       carla_actor=carla_actor,
                                       parent=parent_actor)
        elif sensor_type == 'sensor.camera.semantic_segmentation':
            sensor_actor = SemanticSegmentationCamera(uid=self.generate_uid(),
                                                      name=sensor_name,
                                                      base_save_dir=parent_actor.get_save_dir(),
                                                      carla_actor=carla_actor,
                                                      parent=parent_actor)
        elif sensor_type == 'sensor.lidar.ray_cast':
            sensor_actor = Lidar(uid=self.generate_uid(),
                                 name=sensor_name,
                                 base_save_dir=parent_actor.get_save_dir(),
                                 carla_actor=carla_actor,
                                 parent=parent_actor)
        elif sensor_type == 'sensor.lidar.ray_cast_semantic':
            sensor_actor = SemanticLidar(uid=self.generate_uid(),
                                         name=sensor_name,
                                         base_save_dir=parent_actor.get_save_dir(),
                                         carla_actor=carla_actor,
                                         parent=parent_actor)
        elif sensor_type == 'sensor.other.radar':
            sensor_actor = Radar(uid=self.generate_uid(),
                                 name=sensor_name,
                                 base_save_dir=parent_actor.get_save_dir(),
                                 carla_actor=carla_actor,
                                 parent=parent_actor)
        else:
            print("Unsupported sensor type: {}".format(sensor_type))
            raise AttributeError
        sensor_node = Node(sensor_actor, NodeType.SENSOR)
        return sensor_node

    def generate_uid(self):
        uid = self._uid_count
        self._uid_count += 1
        return uid
