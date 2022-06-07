#!/usr/bin/python3
import os

import carla
from recorder.actor_factory import ActorFactory, Node
from multiprocessing.dummy import Pool as ThreadPool


class ActorTree(object):
    def __init__(self, world: carla.World, actor_config_file=None, base_save_dir=None):
        self.world = world
        self.actor_config_file = actor_config_file
        self.actor_factory = ActorFactory(self.world, base_save_dir)
        self.root = Node(None)
        self.node_list = []

    def init(self):
        self.root = self.actor_factory.create_actor_tree(self.actor_config_file)
        self.node_list.append(self.root)
        for node in self.root.get_children():
            self.node_list.append(node)
            for sensor_node in node.get_children():
                self.node_list.append(sensor_node)

    def destroy(self):
        self.root.destroy()

    def add_node(self, node):
        self.root.add_child(node)

    def tick_controller(self):
        for v2i_layer_node in self.root.get_children():
            v2i_layer_node.tick_controller()

    def tick_data_saving(self, frame_id, timestamp: float):
        thread_pool = ThreadPool()
        frame_id_list = [frame_id for i in range(len(self.node_list))]
        timestamp_list = [timestamp for i in range(len(self.node_list))]
        thread_pool.starmap_async(self.save_data, zip(frame_id_list, timestamp_list, self.node_list))
        thread_pool.close()
        thread_pool.join()

    def save_data(self, frame_id, timestamp: float, node: Node):
        node.tick_data_saving(frame_id, timestamp)

    def print_tree(self):
        print("------ Actor Tree BEGIN ------")
        for node in self.root.get_children():
            print("- {}".format(node.get_actor().name))
            for child_node in node.get_children():
                if child_node is not None:
                    print("|- {}".format(child_node.get_actor().name))
        print("------ Actor Tree END ------")
