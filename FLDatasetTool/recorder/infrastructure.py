#!/usr/bin/python3
import carla
from recorder.actor import PseudoActor


class Infrastructure(PseudoActor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 transform: carla.Transform):
        super().__init__(uid=uid, name=name, parent=None)
        self.carla_transform = transform
        self.save_dir = '{}/{}'.format(base_save_dir, self.name)

    def get_carla_transform(self):
        return self.carla_transform

    def get_carla_bbox(self):
        return carla.BoundingBox(self.carla_transform.location,
                                 carla.Vector3D(1.0, 1.0, 1.0))

    def get_type_id(self):
        return 'others.infrastructure'

    def get_save_dir(self):
        return self.save_dir

    def save_to_disk(self, frame_id, timestamp, debug=False):
        if debug:
            print("\tInfrastructure status recorded: uid={} name={}".format(self.uid, self.name))


