#!/usr/bin/env python3
import os
import sys
import time
import glob
# amend relative import
from pathlib import Path
sys.path.append( Path(__file__).resolve().parent.parent.as_posix() ) #repo path
sys.path.append( Path(__file__).resolve().parent.as_posix() ) #file path
from params import *

try:
    _egg_file = sorted(Path(CARLA_PATH, 'PythonAPI/carla/dist').expanduser().glob('carla-*%d.*-%s.egg'%(
        sys.version_info.major,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    )))[0].as_posix()
    sys.path.append(_egg_file)
except IndexError:
    print('CARLA Egg File Not Found.')
    exit()

import carla
import random
import weakref
import logging
import argparse
# import mayavi.mlab

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
from carla import ColorConverter as cc

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
ApplyVehicleControl = carla.command.ApplyVehicleControl
Attachment = carla.AttachmentType

import matplotlib.pyplot as plt     
from mpl_toolkits.mplot3d import Axes3D

try:
    sys.path.append(Path(CARLA_PATH, 'PythonAPI/carla').expanduser().as_posix() )
    sys.path.append(Path(CARLA_PATH, 'PythonAPI/examples').expanduser() )
except IndexError:
    pass

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

from utils.get2Dlabel import ClientSideBoundingBoxes

class Args(object):
    def __init__(self, argv=None):
        # client information
        self.host = '127.0.0.1'
        self.port = 2000
        self.tm_port = 8000
        self.time_out = 10.0
        self.fixed_delta_seconds = 0.05

        # map information
        self.map_name = TOWN_MAP
        # self.spectator_point = [(-44.30713653564453, -59.78025817871094,100),(-90,90,0)]
        # self.spectator_point = [(-38.54999923706055, 11.59999942779541, 100),(-90,90,0)]
        self.spectator_point = [(50,-130,180),(-90,90,0)]
        

        self.ROI = [[-140,-150],[-140, 140],[150, 140],[150, -150]]
        self.initial_spawn_ROI = [[-80,-70,0,100],
                                    [-90,-80,-120,0],
                                    [-150,-90,0,10],
                                    [0,10,0,100],
                                    [-10,0,-150,0],
                                    [0,150,-10,0]]
        self.additional_spawn_ROI = [[-80,-70,90,100],
                                    [-90,-80,-120,-100],
                                    [-150,-140,0,10],
                                    [0,10,80,135],
                                    [-10,0,-150,-120],
                                    [140,150,-10,0]]

        # server information
        # self.fixed_delta_seconds = 0.1
        # agent information
        self.agent = 'BehaviorAgent'
        # record information
        self.sync = True
        self.time = time.strftime("%Y_%m%d_%H%M", time.localtime())
        self.recorder_filename = ( LOG_PATH / ('record'+self.time+'.log') ).as_posix()
        # self.recorder_filename = os.getcwd() + '/' + 'log/record' + self.time + '.log'
        if argv:
            self.task = argv[1]
            if self.task == 'record' and len(argv) >= 4:
                self.hd_id = [int(id) for id in argv[2].split(',')] #'hd' for 'human driver'
                self.av_id = [int(id) for id in argv[3].split(',')] #'av' for 'autonomous vehicle'
            if len(argv) > 2 and self.task == 'replay':
                self.recorder_filename = self.recorder_filename[:-13] + sys.argv[2] + '.log'
            elif len(argv) == 2 and self.task == 'replay':
                record_path = os.listdir(self.recorder_filename[:-24])
                self.recorder_filename = self.recorder_filename[:-24] + record_path[-1]
                
        self.time_factor = 1.0
        self.camera = 0
        self.start = 0
        self.duration = 0
        # raw data information
        self.raw_data_path = RAW_DATA_PATH / ('record'+self.time)
        # self.raw_data_path = 'tmp/record' + self.time + '/'
        self.image_width = 1242
        self.image_height = 375
        self.VIEW_FOV = 90
        
        self.calibration = np.identity(3)
        self.calibration[0, 2] = self.image_width / 2.0
        self.calibration[1, 2] = self.image_height / 2.0
        self.calibration[0, 0] = self.calibration[1, 1] = self.image_width / (2.0 * np.tan(self.VIEW_FOV * np.pi / 360.0))

        self.sample_frequence = 1  # 20 frames/s
        # self.sample_frequence = 10  # 2 frames/s

class Map(object):
    def __init__(self, args):
        self.pretrain_model = True
        self.client = carla.Client(args.host, args.port)
        self.world = self.client.get_world()
        self.initial_spectator(args.spectator_point)
        self.tmp_spawn_points = self.world.get_map().get_spawn_points()
        if self.pretrain_model:
            self.initial_spawn_points =self.tmp_spawn_points
        else:
            self.initial_spawn_points = self.check_spawn_points(args.initial_spawn_ROI)
        self.additional_spawn_points = self.check_spawn_points(args.additional_spawn_ROI)
        self.destination = self.init_destination(self.tmp_spawn_points, args.ROI)
        self.ROI = args.ROI
        try:
            self.hd_id = args.hd_id
            self.av_id = args.av_id
        except:
            self.hd_id = []
            self.av_id = []

    def initial_spectator(self, spectator_point):
        spectator = self.world.get_spectator()
        spectator_point_transform = carla.Transform(carla.Location( spectator_point[0][0],
                                                                    spectator_point[0][1],
                                                                    spectator_point[0][2]), 
                                                    carla.Rotation( spectator_point[1][0],
                                                                    spectator_point[1][1],
                                                                    spectator_point[1][2]))
        spectator.set_transform(spectator_point_transform)

    def check_spawn_points(self, check_spawn_ROI):
        tmp_spawn_points = []
        tmpx,tmpy = [],[]
        for tmp_transform in self.tmp_spawn_points:
            tmp_location = tmp_transform.location
            for edge in check_spawn_ROI:
                if tmp_location.x > edge[0] and tmp_location.x < edge[1] and tmp_location.y > edge[2] and tmp_location.y < edge[3]:
                    tmp_spawn_points.append(tmp_transform)
                    tmpx.append(tmp_location.x)
                    tmpy.append(tmp_location.y)
                    continue
        # self.plot_points(tmpx,tmpy)
        return tmp_spawn_points
    
    def plot_points(self,tmpx,tmpy):
        plt.figure(figsize=(8,7))
        ax = plt.subplot(111)
        ax.axis([-50,250,50,350])
        ax.scatter(tmpx,tmpy)
        for index in range(len(tmpx)):
            ax.text(tmpx[index],tmpy[index],index)
        plt.show()

    def plot_lights(self,tmpx,tmpy):
        plt.figure(figsize=(8,7))
        ax = plt.subplot(111)
        ax.axis([-300,300,-300,300])
        ax.scatter(tmpx,tmpy, marker='o', c='r')
        for index in range(len(tmpx)):
            ax.text(tmpx[index],tmpy[index],index)
        plt.show()

    def init_destination(self, spawn_points, ROI):
        destination = []
        tmpx,tmpy = [],[]
        for p in spawn_points:
            if not self.inROI([p.location.x, p.location.y], ROI):
                destination.append(p)
                tmpx.append(p.location.x)
                tmpy.append(p.location.y)
        # self.plot_points(tmpx,tmpy)
        return destination
    
    def sign(self,a,b,c):
        return (a[0]-c[0])*(b[1]-c[1])-(b[0]-c[0])*(a[1]-c[1])
    
    def inROI(self,x,ROI):
        d1=self.sign(x,ROI[0],ROI[1])
        d2=self.sign(x,ROI[1],ROI[2])
        d3=self.sign(x,ROI[2],ROI[3])
        d4=self.sign(x,ROI[3],ROI[0])

        has_neg=(d1<0) or (d2<0) or (d3<0) or (d4<0)
        has_pos=(d1>0) or (d2>0) or (d3>0) or (d4>0)
        return not(has_neg and has_pos)

    def shuffle_spawn_points(self, spawn_points, start=False):
        # random.shuffle(spawn_points)
        if self.pretrain_model:
            # self.av_id = [4,5,27,20,97,22,14,77,47]
            # self.hd_id = [19,21,29,31,44,48,87,96] + [i for i in range(50,70)]

            cav = [spawn_points[i] for i in self.av_id]
            hd = [spawn_points[i] for i in self.hd_id]
            if len(cav) == 0 and len(hd) == 0:
                return spawn_points[:60],spawn_points[-20:]
            else:
                return hd, cav
            
class Server(object):
    def __init__(self):
        pass

class Vehicle_Agent(BehaviorAgent):
    def __init__(self, vehicle):
        BehaviorAgent.__init__(self, vehicle)
    
    def planning_ang_control(self):
        pass

class CAVcontrol_Thread(Thread):   
    #   继承父类threading.Thread
    def __init__(self, vehicle, world, destination, num_min_waypoints, control):
        Thread.__init__(self)
        self.v = vehicle
        self.w = world
        self.d = destination
        self.n = num_min_waypoints
        self.c = control
        self.start()   

    def run(self):                      
        #   把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        self.control = None
        self.v.update_information(self.w)
        if len(self.v.get_local_planner().waypoints_queue) < self.n:
            self.v.reroute(self.d)
        speed_limit = self.v.vehicle.get_speed_limit()
        self.v.get_local_planner().set_speed(speed_limit)
        self.control = self.c(self.v.vehicle.id, self.v.run_step())

    def return_control(self):           
        #   threading.Thread.join(self) # 等待线程执行完毕
        self.join()
        try:
            return self.control
        except Exception:
            print('This is an issue')
   
class CAVcollect_Thread(Thread):
    def __init__(self, parent_id, sensor_attribute, sensor_transform, args):
        Thread.__init__(self)
        self.recording = False
        self.args = args
        gamma_correction = 2.2
        Attachment = carla.AttachmentType
        self.client = carla.Client(self.args.host, self.args.port)
        world = self.client.get_world()
        self.sensor = None
        self._parent = world.get_actor(parent_id)
        self._camera_transforms = sensor_transform#(sensor_transform, Attachment.Rigid)
        bp_library = world.get_blueprint_library()
        bp = bp_library.find(sensor_attribute[0])

        if "vehicle"  in self._parent.type_id:         
            if sensor_attribute[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(self.args.image_width))
                bp.set_attribute('image_size_y', str(self.args.image_height))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in sensor_attribute[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif sensor_attribute[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '100')
                bp.set_attribute('channels','16')
                bp.set_attribute('points_per_second','2240000')
                bp.set_attribute('rotation_frequency','20')
                bp.set_attribute('sensor_tick', str(0.05))
                bp.set_attribute('dropoff_general_rate', '0.0')
                bp.set_attribute('dropoff_intensity_limit', '1.0')
                bp.set_attribute('dropoff_zero_intensity', '0.0')
            sensor_attribute.append(bp)
            self.sensor_attribute = sensor_attribute

        if "traffic"  in self._parent.type_id:         
            if sensor_attribute[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(self.args.image_width))
                bp.set_attribute('image_size_y', str(self.args.image_height))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in sensor_attribute[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif sensor_attribute[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '100')
                bp.set_attribute('channels','64')
                bp.set_attribute('points_per_second','2240000')
                bp.set_attribute('rotation_frequency','20')
                bp.set_attribute('sensor_tick', str(0.05))
                bp.set_attribute('dropoff_general_rate', '0.0')
                bp.set_attribute('dropoff_intensity_limit', '1.0')
                bp.set_attribute('dropoff_zero_intensity', '0.0')
            sensor_attribute.append(bp)
            self.sensor_attribute = sensor_attribute

    
    def run(self):
        self.set_sensor()
    
    def set_sensor(self):
        self.sensor = self._parent.get_world().spawn_actor(
            self.sensor_attribute[-1],
            self._camera_transforms[0],
            attach_to=self._parent)
            # attachment_type=self._c#amera_transforms[1])
        filename = Path(self.args.raw_data_path,
                        '%s_%d'%(self._parent.type_id, self._parent.id),
                        '%s_%d'%(self.sensor.type_id, self.sensor.id)
                    ).as_posix()

        if 'traffic' in self._parent.type_id:
            filename = Path(self.args.raw_data_path,
                        '%s_%d'%('road.sensor.lidar', self._parent.id),
                        '%s_%d'%(self.sensor.type_id, self.sensor.id)
                    ).as_posix()
        
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CAVcollect_Thread._parse_image(weak_self, image, filename))
        # self.sensor.stop()
        # print(filename)    
        

    def get_sensor_id(self):
        self.join()
        return self.sensor.id

        
    @staticmethod
    def _parse_image(weak_self, image, filename):
        self = weak_self()
        if image.frame % self.args.sample_frequence != 0:
            return
        if self.sensor.type_id.startswith('sensor.camera'):
            image.convert(self.sensor_attribute[1])
            image.save_to_disk(filename+'/%010d' % image.frame)
        else:
            image.save_to_disk(filename+'/%010d' % image.frame)
        
class Scenario(object):
    def __init__(self, args):
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.time_out)
        self.world = self.client.load_world(args.map_name)
        self.traffic_manager = self.client.get_trafficmanager(args.tm_port)
        self.map = Map(args)
        self.recording_rawdata = False
        # agent information
        self.HD_blueprints = self.world.get_blueprint_library().filter('vehicle.audi.a2')
        self.CAV_blueprints = self.world.get_blueprint_library().filter('vehicle.tesla.model3')
        # sensor information
        self.sensor_attribute = [['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
                                ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}]]
        self.sensor_transform = [(carla.Transform(carla.Location(x=0, z=1.7)), Attachment.Rigid),
                                (carla.Transform(carla.Location(x=0, z=1.7)), Attachment.Rigid)]
        self.args = args
        weak_self = weakref.ref(self)
        self.world.on_tick(lambda world_snapshot: self.on_world_tick(weak_self,world_snapshot))

    @staticmethod
    def parse_transform(transform):
        return [transform.location.x,transform.location.y,transform.location.z,transform.rotation.roll,transform.rotation.pitch,transform.rotation.yaw]

    @staticmethod
    def parse_bounding_box(bounding_box):
        return [bounding_box.extent.x,bounding_box.extent.y,bounding_box.extent.z, bounding_box.location.z]

    @staticmethod
    def on_world_tick(weak_self, world_snapshot):
        self = weak_self()  
        if world_snapshot.frame % self.args.sample_frequence != 0:
            return
        if self.args.task == 'replay':# or world_snapshot.frame % 2 == 0:
            return
        actors = self.world.get_actors()
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        # lights_list = list([4,7])
        lights_list = [int(id) for id in sys.argv[4].split(',')]
        vehicles, sensors, CAV_vehicles = [], [], []
        for actor in actors:
            str_actor = [str(actor.type_id),actor.id] + Scenario.parse_transform(actor.get_transform())
            # if 'lidar' in actor.type_id or 'rgb' in actor.type_id:
            #     print(str(actor.type_id),actor.get_transform().rotation.pitch,actor.get_transform().rotation.roll)
            if 'vehicle' in actor.type_id:
                str_actor += Scenario.parse_bounding_box(actor.bounding_box)
                vehicles.append(str_actor)
            elif 'sensor' in actor.type_id:
                str_actor += [0,0,0] + [actor.parent.id]
                sensors.append(str_actor)
            elif 'traffic' in actor.type_id:
                for light_id in lights_list:
                    if actor.id == traffic_lights[light_id].id:
                        str_actor = ['road.sensor.lidar',actor.id] + Scenario.parse_transform(actor.get_transform())
                        str_actor += Scenario.parse_bounding_box(actor.bounding_box)
                        vehicles.append(str_actor)


        actors = np.array(vehicles + sensors)
        _label_path = Path(self.args.raw_data_path, 'label')
        _label_path.mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(self.args.raw_data_path+'label'):
        #     os.makedirs(self.args.raw_data_path+'label')
        if len(actors) != 0:
            _filename = ( _label_path / ('%010d.txt'%(world_snapshot.frame)) ).as_posix()
            np.savetxt(_filename, actors, fmt='%s', delimiter=' ')
            # np.savetxt(self.args.raw_data_path + '/label/%010d.txt' % world_snapshot.frame, actors, fmt='%s', delimiter=' ')

        # 2D bounding box
        vehicles = self.world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if str(vehicle.id) in self.sensor_relation.keys():
                sensor_list = self.sensor_relation[str(vehicle.id)]
            else:
                continue
            calib_info = []
            for sensor_id in sensor_list:
                sensor = self.world.get_actor(sensor_id)
                if 'lidar' in sensor.type_id:
                    lidar = self.world.get_actor(sensor_id)

            for sensor_id in sensor_list:
                sensor = self.world.get_actor(sensor_id)
                if 'rgb' in sensor.type_id:
                    sensor.calibration = self.args.calibration
                    tmp_bboxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles,sensor)
                    image_label_path = Path(self.args.raw_data_path,
                                            vehicle.type_id + '_' + str(vehicle.id),
                                            sensor.type_id + '_' + str(sensor.id)
                                        ).as_posix()
                    # image_label_path = self.args.raw_data_path + \
                    #                     vehicle.type_id + '_' + str(vehicle.id) + '/' + \
                    #                     sensor.type_id + '_' + str(sensor.id)
                    
                    if not os.path.exists(image_label_path+'_label'):
                        os.makedirs(image_label_path+'_label')
                    if len(tmp_bboxes) != 0:
                        np.savetxt(image_label_path+'_label/%010d.txt' % world_snapshot.frame, tmp_bboxes, fmt='%s', delimiter=' ')
                    # lidar_to_camera_matrix = ClientSideBoundingBoxes.get_lidar_to_camera_matrix(lidar, sensor)
                    # calib_info.append(lidar_to_camera_matrix)

            # 2D bounding box for road sensors
            traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
            for light in traffic_lights:
                if str(light.id) in self.sensor_relation.keys():
                    sensor_list = self.sensor_relation[str(light.id)]
                else:
                    continue
                calib_info = []
                for sensor_id in sensor_list:
                    sensor = self.world.get_actor(sensor_id)
                    if 'lidar' in sensor.type_id:
                        lidar = self.world.get_actor(sensor_id)

                for sensor_id in sensor_list:
                    sensor = self.world.get_actor(sensor_id)
                    if 'rgb' in sensor.type_id:
                        sensor.calibration = self.args.calibration
                        tmp_bboxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles,sensor)
                        image_label_path = Path(self.args.raw_data_path,
                                            'road.sensor.lidar' + '_' + str(light.id),
                                            sensor.type_id + '_' + str(sensor.id)
                                        ).as_posix()
                    
                        if not os.path.exists(image_label_path+'_label'):
                            os.makedirs(image_label_path+'_label')
                        if len(tmp_bboxes) != 0:
                            np.savetxt(image_label_path+'_label/%010d.txt' % world_snapshot.frame, tmp_bboxes, fmt='%s', delimiter=' ')


    def look_for_spawn_points(self, args):
        try:
            self.start_look(args)
            if not args.sync or not self.synchronous_master:
                self.world.wait_for_tick()
            else:
                start = self.world.tick()
            while True:
                if args.sync and self.synchronous_master:
                    now = self.run_step()
                    if (now - start) % 1000 == 0:
                        print('Frame ID:'+str(now))

                else:
                    self.world.wait_for_tick()
        finally:
            try:
                print('stop from frameID: %s.' % now)
            finally:
                pass
            self.stop_look(args)
            pass
    
    def start_look(self, args):
        self.synchronous_master = False
        if args.sync:
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                self.synchronous_master = True
                self.traffic_manager.set_synchronous_mode(True)
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = args.fixed_delta_seconds
                self.world.apply_settings(settings)
            else:
                self.synchronous_master = False
                print('synchronous_master is False.')
        tmpx,tmpy = [],[]
        for tmp_transform in self.map.initial_spawn_points:
            tmp_location = tmp_transform.location
            tmpx.append(((tmp_location.x-100)*-1)+100)
            tmpy.append(tmp_location.y)
        self.map.plot_points(tmpx,tmpy)

        # traffic lights
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')

        tmpx,tmpy = [],[]
        for tmp_light in traffic_lights:
            tmp_location = tmp_light.get_transform().location
            tmpx.append(((tmp_location.x-100)*-1)+100)
            tmpy.append(tmp_location.y)
        self.map.plot_lights(tmpx,tmpy)


    def stop_look(self, args):
        print(args.sync)
        if args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        self.world = self.client.reload_world()
        self.map.initial_spectator(args.spectator_point)

    def generate_data(self, args):
        self.recording_rawdata = True
        try:
            self.start_record(args)
            if not args.sync or not self.synchronous_master:
                self.world.wait_for_tick()
            else:
                start = self.world.tick()  
                if self.dynamic_weather:
                    self.weather.tick(1)
                    self.world.set_weather(self.weather.weather)
                print('start from frameID: %s.' % start)
            while True:
                
                if args.sync and self.synchronous_master:
                    time.sleep(1)
                    now = self.run_step()
                    # if (now - start) % 1000 == 0:
                    if (now - start) % 10 == 0:
                        print('Frame ID:'+str(now))
                    if (now - start) == 200:
                        print('finish!')
                        sys.exit()
                else:
                    self.world.wait_for_tick()
        finally:
            try:
                print('stop from frameID: %s.' % now)
            finally:
                pass
            self.stop_record(args)
            pass

    def start_record(self, args):
        self.synchronous_master = False
        self.dynamic_weather = False
        
        # change the weather
        weather = self.world.get_weather()
        # weather.precipitation_deposits = 80
        # weather.precipitation = 80
        # weather.cloudiness = 80.0
        # weather.sun_altitude_angle = -30
        self.world.set_weather(weather)


        if self.dynamic_weather:
            from dynamic_weather import Weather
            w = self.world.get_weather()
            w.precipitation = 80
            weather = Weather(w)
            self.weather = weather
        if args.sync:
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                self.synchronous_master = True
                self.traffic_manager.set_synchronous_mode(True)
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = args.fixed_delta_seconds
                self.world.apply_settings(settings)
            else:
                self.synchronous_master = False
                print('synchronous_master is False.')
        # if not os.path.exists('log'):
        #     os.mkdir('log')
        #     print('mkdir log finished.')
        print("Recording on file: %s" % self.client.start_recorder(args.recorder_filename))
        self.agent_list = []
        self.infrastructure_list = []
        self.sensor_relation = {}
        self.sensor_thread = []
        HD_spawn_points, CAV_spawn_points = self.map.shuffle_spawn_points(self.map.initial_spawn_points,start=True)
        # print(len(CAV_spawn_points))
        self.HD_agents = self.spawn_actorlist('vehicle', self.HD_blueprints, HD_spawn_points)
        print(len(self.HD_agents))
        self.CAV_agents = self.spawn_actorlist('vehicle', self.CAV_blueprints, CAV_spawn_points)
        lights = [int(id) for id in sys.argv[4].split(',')]
        self.infrastructure = self.spawn_actorlist('infrastructure', 0, lights)

    def stop_record(self, args):
        if args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.CAV_agents+self.HD_agents])
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.camera_list])
        print('\ndestroying %d vehicles' % len(self.CAV_agents+self.HD_agents))
        self.sensor_list = []
        for sensor in self.sensor_relation.values():
            self.sensor_list += sensor
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
        print('\ndestroying %d sensors' % len(self.sensor_list))
        self.client.stop_recorder()
        print("Stop recording")
    
    def spawn_actorlist(self, actor_type, agent_blueprint=None, spawn_points=None, parent_agent=None):
        bacth_spawn = []
        id_list = []
        if actor_type == 'vehicle':
            
            if not random.choice(agent_blueprint).id.startswith('vehicle.tesla'):
                # HD_agents
                # print(len(spawn_points))
                for n, transform in enumerate(spawn_points):
                        blueprint = random.choice(agent_blueprint)
                        while 'tesla' in blueprint.id or 'crossbike' in blueprint.id or 'low_rider' in blueprint.id:
                            blueprint = random.choice(agent_blueprint)
                        bacth_spawn.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
                for response in self.client.apply_batch_sync(bacth_spawn, False):
                    if response.error:
                        print(response.error)
                        logging.error(response.error)
                    else:
                        tmp_vehicle = self.world.get_actor(response.actor_id)
                        if tmp_vehicle.bounding_box.extent.y < 0.6:
                            print(tmp_vehicle.bounding_box.extent.y)
                            tmp_vehicle.destroy()
                        # print(tmp_vehicle.bounding_box.extent.y)
                        if int(tmp_vehicle.attributes['number_of_wheels']) == 2:
                            tmp_vehicle.destroy()
                        else:
                            id_list.append(response.actor_id)
            elif random.choice(agent_blueprint).id.startswith('vehicle.tesla'):
                # CAV_agents
                for n, transform in enumerate(spawn_points):
                        blueprint = random.choice(agent_blueprint)
                        bacth_spawn.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))
                for response in self.client.apply_batch_sync(bacth_spawn, True):
                    if response.error:
                        logging.error(response.error)
                    else:
                        id_list.append(response.actor_id)
                        vehicle = self.client.get_world().get_actor(response.actor_id)
                        tmp_sensor_id_list = self.spawn_actorlist('sensor',self.sensor_attribute,self.sensor_transform,response.actor_id)
                        self.sensor_relation[str(response.actor_id)] = tmp_sensor_id_list
                        random.shuffle(self.map.destination)
                        tmp_agent = Vehicle_Agent(vehicle)
                        # tmp_agent.set_destination(tmp_agent.vehicle.get_location(), self.map.destination[0].location, clean=True) # <= carla 0.9.11
                        tmp_agent.set_destination(self.map.destination[0].location)  # >= carla 0.9.12
                        self.agent_list.append(tmp_agent)
        
        elif actor_type == 'sensor':
            # sensor agents
            for index in range(len(self.sensor_attribute)):
                sensor_attribute = self.sensor_attribute[index]
                transform = self.sensor_transform[index]
                tmp_sensor = CAVcollect_Thread(parent_agent, sensor_attribute, transform, self.args)
                tmp_sensor.start()
                self.sensor_thread.append(tmp_sensor)
                id_list.append(tmp_sensor.get_sensor_id())

        elif actor_type == 'road_sensor':
            # sensor agents
            for index in range(len(self.sensor_attribute)):
                sensor_attribute = self.sensor_attribute[index]
                road_sensor_transform = [(carla.Transform(carla.Location(x=0, z=1.7),carla.Rotation(pitch=0,yaw=90)), Attachment.Rigid),
                                (carla.Transform(carla.Location(x=0, z=1.7),carla.Rotation(pitch=0,yaw=90)), Attachment.Rigid)]

                # road_sensor_transform = [(carla.Transform(carla.Location(x=0, z=1.7),carla.Rotation(pitch=0,yaw=0)), Attachment.Rigid),
                #                 (carla.Transform(carla.Location(x=0, z=1.7),carla.Rotation(pitch=0,yaw=0)), Attachment.Rigid)]

                tmp_sensor = CAVcollect_Thread(parent_agent, sensor_attribute, road_sensor_transform[index], self.args)
                tmp_sensor.start()
                self.sensor_thread.append(tmp_sensor)
                id_list.append(tmp_sensor.get_sensor_id())
        
        elif actor_type == 'infrastructure':
            # infrastructure sensors
            for n, point in enumerate(spawn_points):
                traffic_light = None
                traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
                traffic_light = traffic_lights[point]
                # debug = self.world.debug
                # #在摄像头图像中显示bbox
                # debug.draw_box(carla.BoundingBox(traffic_light.get_transform().location, carla.Vector3D(5, 5, 5)),
                #        traffic_light.get_transform().rotation, 1, carla.Color(255, 0, 0, 0), 0)
                tmp_sensor_id_list = self.spawn_actorlist('road_sensor', None, None, traffic_light.id)
                self.sensor_relation[str(traffic_light.id)] = tmp_sensor_id_list
                self.infrastructure_list.append(traffic_light)

        return id_list

    def check_vehicle_state(self):
        for v_id in self.HD_agents:

            vehicle = self.world.get_actor(v_id)
            v_position = vehicle.get_transform().location
            if not(self.map.inROI([v_position.x, v_position.y], self.map.ROI)):
                vehicle.destroy()
                self.HD_agents.remove(v_id)
        
        for v_id in self.CAV_agents:
            vehicle = self.world.get_actor(v_id)
            v_position = vehicle.get_transform().location
            if not(self.map.inROI([v_position.x, v_position.y], self.map.ROI)):
                # for agent in self.agent_list:
                #     if agent.vehicle.id == v_id:
                #         self.agent_list.remove(agent)
                #         delete camera
                #         break
                for sensor_id in self.sensor_relation[str(v_id)]:
                    sensor = self.world.get_actor(sensor_id)
                    if sensor.is_listening:
                        print(sensor.id)
                        sensor.stop()
                    sensor.destroy()
                self.sensor_relation.pop(str(v_id))
                vehicle.destroy()
                self.CAV_agents.remove(v_id)

    def run_step(self):        
        # batch_control = []
        # thread_list = []
        # num_min_waypoints = 21
        # for agent in self.agent_list:
        #     t = CAVcontrol_Thread(agent, self.world, self.map.destination, num_min_waypoints, ApplyVehicleControl)
        #     thread_list.append(t)
        # for t in thread_list:
        #     batch_control.append(t.return_control())
        # for response in self.client.apply_batch_sync(batch_control, False):
        #     if response.error:
        #         logging.error(response.error)
        if not self.map.pretrain_model:
            self.check_vehicle_state()
        return self.world.tick()

    def add_anget_and_vehicles(self):
        HD_additional_spawn_points,CAV_additional_spawn_points = self.map.shuffle_spawn_points(self.map.additional_spawn_points)
        self.HD_agents += self.spawn_actorlist('vehicle', self.HD_blueprints, HD_additional_spawn_points)
        self.CAV_agents += self.spawn_actorlist('vehicle', self.CAV_blueprints, CAV_additional_spawn_points)

    def return_cor(self, waypoint):
        location = waypoint.transform.location
        return [location.x,location.y,location.z]

    def get_road(self, world_map):
        WAYPOINT_DISTANCE = 10
        topology = world_map.get_topology()
        road_list = []
        for wp_pair in topology:
            current_wp = wp_pair[0]
            # Check if there is a road with no previus road, this can happen
            # in opendrive. Then just continue.
            if current_wp is None:
                continue
            # First waypoint on the road that goes from wp_pair[0] to wp_pair[1].
            current_road_id = current_wp.road_id
            wps_in_single_road = [self.return_cor(current_wp)]
            # While current_wp has the same road_id (has not arrived to next road).
            while current_wp.road_id == current_road_id:
                # Check for next waypoints in aprox distance.
                available_next_wps = current_wp.next(WAYPOINT_DISTANCE)
                # If there is next waypoint/s?
                if available_next_wps:
                    # We must take the first ([0]) element because next(dist) can
                    # return multiple waypoints in intersections.
                    current_wp = available_next_wps[0]
                    wps_in_single_road.append(self.return_cor(current_wp))
                else: # If there is no more waypoints we can stop searching for more.
                    break
            pcd1 = o3d.geometry.PointCloud()
            pdc2 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(wps_in_single_road[:-1])
            pdc2.points = o3d.utility.Vector3dVector(wps_in_single_road[1:])
            corr = [(i,i+1) for i in range(len(wps_in_single_road)-2)]
            lineset = o3d.geometry.LineSet.create_from_point_cloud_correspondences(pcd1,pdc2,corr)
            lineset.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
            road_list.append(lineset)
        return road_list

    def find_and_replay(self, args):
        self.recording_rawdata = True
        # road_net = self.get_road(self.world.get_map())
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
        # o3d.visualization.draw_geometries(road_net+[mesh],height=1280,width=1920)
        try:
            start_time = time.time()
            replay_time = self.start_replay(args)
            while time.time()-start_time < replay_time:
                self.world.tick()
        finally:
            print('stop replay...')
            time.sleep(2)
            self.stop_replay(args)
            pass

    def start_replay(self, args):
        settings = self.world.get_settings()        
        settings.synchronous_mode = True        
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        self.world.apply_settings(settings)

         # set the time factor for the replayer
        self.client.set_replayer_time_factor(args.time_factor)

         # replay the session
        output = self.client.replay_file(args.recorder_filename, args.start, args.duration, args.camera)
        replay_time = self.find_replay_time(output, args.duration)
        print('start replay...{}'.format(str(output)))
        return replay_time

    def stop_replay(self, args):
        actor_list =[]
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor_list.append(actor.id)
        self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        if self.args.sync:# and synchronous_master:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        print('destroying %d vehicles' % len(actor_list))
        self.world = self.client.reload_world()
        self.map.initial_spectator(args.spectator_point)
        exit()

    def find_replay_time(self, output, duration):
        index_start = output.index('-')+2
        index_end = output.index('(')-2
        total_time = float(output[index_start:index_end])
        if duration == 0:
            return total_time
        else:
            return duration


if __name__ == "__main__":

    args = Args(sys.argv)
    scenario = Scenario(args)
    if args.task == 'spawn':
        scenario.look_for_spawn_points(args)
    elif args.task == 'record':
        scenario.generate_data(args)
    elif args.task == 'replay':
        scenario.find_and_replay(args)
