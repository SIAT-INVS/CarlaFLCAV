#!/usr/bin/python3
import copy
import csv
import os
import weakref
from queue import Queue

import carla

from recorder.actor import Actor


class Sensor(Actor):
    def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 parent,
                 carla_actor: carla.Sensor):
        super(Sensor, self).__init__(uid=uid, name=name, parent=parent, carla_actor=carla_actor)
        self.sensor_type = copy.deepcopy(self.get_type_id())
        self.save_dir = base_save_dir + '/{}'.format(name)
        self.queue = Queue()
        weak_self = weakref.ref(self)
        self.carla_actor.listen(lambda sensor_data: Sensor.data_callback(weak_self,
                                                                         sensor_data,
                                                                         self.queue))

        self.csv_fieldnames = ['frame',
                               'timestamp',
                               'x', 'y', 'z',
                               'roll', 'pitch', 'yaw']
        self._first_frame = True

    @staticmethod
    def data_callback(weak_self, sensor_data, data_queue: Queue):
        data_queue.put(sensor_data)

    def save_to_disk(self, frame_id, timestamp,  debug=False):
        sensor_frame_id = 0
        while sensor_frame_id < frame_id:
            sensor_data = self.queue.get(True, 1.0)
            sensor_frame_id = sensor_data.frame

            # Drop previous data
            if sensor_frame_id < frame_id:
                continue

            # make sure target path exist
            os.makedirs(self.save_dir, exist_ok=True)

            success = self.save_to_disk_impl(self.save_dir, sensor_data)

            if not success:
                print("Save to disk failed!")
                raise IOError

            self.save_pose(frame_id, timestamp)
            self._first_frame = False

            if debug:
                self.print_debug_info(sensor_data.frame, sensor_data)

    def save_to_disk_impl(self, save_dir, sensor_data) -> bool:
        raise NotImplementedError

    def print_debug_info(self, data_frame_id, sensor_data):
        print("\t\tParent uid: {}, Frame: {} uid={} data: {}".format(self.parent.uid, data_frame_id, self.uid, sensor_data))

    def get_save_dir(self):
        return self.save_dir

    def is_first_frame(self):
        return self._first_frame

    def save_pose(self, frame_id, timestamp):
        trans = self.get_transform()
        pose_dict = trans.to_dict()
        pose_dict.update({'frame': frame_id,
                          'timestamp': timestamp})

        csv_path = '{}/poses.csv'.format(self.save_dir)
        if self.is_first_frame():
            with open(csv_path, 'w', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
                writer.writeheader()

        with open(csv_path, 'a', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
            writer.writerow(pose_dict)
