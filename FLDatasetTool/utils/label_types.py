#!/usr/bin/python3

from dataclasses import dataclass
from utils.geometry_types import *


@dataclass
class ObjectLabel(object):
    frame: int
    timestamp: float
    label_type: str
    carla_id: str
    transform: Transform
    bounding_box: BoundingBox

    def __str__(self):
        return "ObjectLabel(frame={}, timestamp={}, label_type={}, carla_id={}, transform={}, bounding_box={}".format(
            self.frame,
            self.timestamp,
            self.label_type,
            self.carla_id,
            self.transform,
            self.bounding_box
        )
