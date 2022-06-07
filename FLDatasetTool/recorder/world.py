#!/usr/bin/python3
import os
import pickle
import carla
from dataclasses import dataclass

from recorder.actor import PseudoActor
from utils.label_types import *
from utils.transform import carla_bbox_to_bbox, carla_transform_to_transform


class WorldActor(PseudoActor):
    def __init__(self, uid, carla_world: carla.World, base_save_dir: str):
        super().__init__(uid, self.get_type_id(), None)
        self.save_dir = "{}/{}_{}".format(base_save_dir, self.get_type_id(), uid)
        self.carla_world = carla_world

    def save_to_disk(self, frame_id, timestamp, debug=False):
        # TODO: Save all object bbox in world
        # Frame Timestamp CityObjectLabel carla_id location rotation box_location box_extent
        object_labels = []

        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Vehicles)
        object_labels += self.get_env_objects_labels(frame_id, timestamp, carla.CityObjectLabel.Pedestrians)

        carla_actors = self.carla_world.get_actors()
        for carla_actor in carla_actors:
            if carla_actor.type_id.startswith('vehicle') \
                    or carla_actor.type_id.startswith('walker'):
                transform = carla_transform_to_transform(carla_actor.get_transform())
                bbox = carla_bbox_to_bbox(carla_actor.bounding_box)
                if carla_actor.type_id.startswith('walker'):
                    label_type = 'pedestrian'
                else:
                    label_type = 'vehicle'
                object_labels.append(ObjectLabel(frame=frame_id,
                                                 timestamp=timestamp,
                                                 label_type=label_type,
                                                 carla_id=carla_actor.id,
                                                 transform=transform,
                                                 bounding_box=bbox))

        if len(object_labels) == 0:
            return False

        os.makedirs(self.save_dir, exist_ok=True)
        with open('{}/{:0>10d}.pkl'.format(self.save_dir, frame_id), 'wb') as pkl_file:
            pickle.dump(obj=object_labels, file=pkl_file)
        if debug:
            print("WorldObjectsLabel: Frame: {} Total counts: {}".format(frame_id, len(object_labels)))
        return True

    def get_type_id(self):
        return 'others.world'

    def get_save_dir(self):
        return self.save_dir

    def get_carla_transform(self) -> carla.Transform:
        return carla.Transform(carla.Location(0, 0, 0), carla.Rotation(0, 0, 0))

    def get_env_objects_labels(self, frame, timestamp, object_type: carla.CityObjectLabel) -> list:
        object_labels = []
        if object_type == carla.CityObjectLabel.Vehicles:
            label_type = 'vehicle'
        elif object_type == carla.CityObjectLabel.Pedestrians:
            label_type = 'pedestrian'
        else:
            label_type = 'any'
        env_objects = self.carla_world.get_environment_objects(object_type=object_type)
        for env_object in env_objects:
            transform = carla_transform_to_transform(env_object.transform)
            bbox_extent = Vector3d(env_object.bounding_box.extent.x,
                                   env_object.bounding_box.extent.y,
                                   env_object.bounding_box.extent.z)
            object_labels.append(ObjectLabel(frame=frame,
                                             timestamp=timestamp,
                                             label_type=label_type,
                                             carla_id=env_object.id,
                                             transform=transform,
                                             bounding_box=BoundingBox(Location(0, 0, 0), bbox_extent)))
        return object_labels
