import os
import sys
import math
import random
import time
import numpy as np
import open3d as o3d
from utils.calibration import Calibration
# print('Please input the legal record path and vehicle id.')
# print('record path: MMDDHHMM    vehicle_id: *')

color = [
        [0.1, 0.9, 0.9],
        [0.1, 0.9, 0.1],
        # [0.9, 0.1, 0.1],
        [0.9,0.9,0.9],
        [0.9, 0.9, 0.1],
        [0.9, 0.1, 0.9],
        [0.9, 0.9, 0.9],
    [0.1, 0.1, 0.9],
    [0.1, 0.9, 0.1],
    # [0.9, 0.9, 0.9],
    [0.9, 0.9, 0.1],
    [0.5, 0.5, 0.5],
    [0.9, 0.9, 0.9],
    [0.5, 0.5, 0.9],
    [0.9, 0.5, 0.1],
    [0.9, 0.5, 0.5],
    [0.9, 0.1, 0.9],
    [0.1, 0.9, 0.9],
]


# color = [
#         [0.1, 0.9, 0.9],
#         # [0.9, 0.1, 0.1],
#         [1,1,1],# 
#         [0.1, 0.9, 0.1],
#         [0.5,0.5,0.5],
#         [0.9, 0.9, 0.1],
#         [0.9, 0.1, 0.9],
#         # [0.9, 0.9, 0.9],
#     ]

color = [
        [0, 0, 0],
        [1, 0, 0],
        # [0.9, 0.1, 0.1],
        [0, 0, 1],# 
    ]

location = [
    [-300, 300, 0],
    [-300, 0, 0],
    [-300, -300, 0],
    [0, -300, 0],
    [0, 300, 0],
    [300, 300, 0],
    [300, 0, 0],
    [300, -300, 0]
]


def get_bboxes(labels):
    bboxes = []
    for label in labels:
        bbox_center = np.array(list(map(float, label[12:15])))
        # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz([0,0,1])
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(
            np.array([0, 0, np.radians(float(label[3]))]))
        bbox_extend = np.array(list(map(float, label[8:11])))
        bbox = o3d.geometry.OrientedBoundingBox(
            bbox_center, bbox_R, bbox_extend)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        lineset.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
        bboxes.append(lineset)
    return bboxes


def deal_bbox_center(bbox_center):
    normal = np.random.normal(loc=0.1, scale=0.9, size=2)
    normal = np.append(normal, 0)
    bbox_center += normal
    return bbox_center


def get_ego_label(ego_label, rotation, location, color=[0.5, 0.5, 0.5]):
    ego_bboxes = []
    for label in ego_label:
        bbox_center = np.array(list(map(float, label[12:15]))) + location
        bbox_center = deal_bbox_center(bbox_center)
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(
            np.array([0, 0, float(label[3])]) + rotation)
        bbox_extend = np.array(list(map(float, label[8:11])))
        bbox = o3d.geometry.OrientedBoundingBox(
            bbox_center, bbox_R, bbox_extend)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        lineset.paint_uniform_color(color)
        ego_bboxes.append(lineset)
    return ego_bboxes


def get_matrix(location, rotation):
    T = np.matrix(np.eye(4))
    T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
    T[0:3, 3] = location.reshape(3, 1)
    return T

def get_ego_ROI(location,ROI,rotation,image_location,color):
    test_lines = o3d.geometry.LineSet()
    test_lines.lines = o3d.utility.Vector2iVector([[0, 1]])
    test_lines.points = o3d.utility.Vector3dVector(np.array([location, image_location-location]))
    test_lines.colors = o3d.utility.Vector3dVector([color])
    x = math.cos(rotation[2])
    y = math.sin(rotation[2])
    vector = np.array([x,y,0]) * 100 + location #-image_location
    line = o3d.geometry.LineSet()
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.points = o3d.utility.Vector3dVector(np.array([location, vector]))
    line.colors = o3d.utility.Vector3dVector([color])
    r = o3d.geometry.get_rotation_matrix_from_xyz([0,0,np.pi/4])
    import copy
    line1 = copy.deepcopy(line.rotate(r,location))
    r1 = o3d.geometry.get_rotation_matrix_from_xyz([0,0,-np.pi/2])
    line2 = copy.deepcopy(line.rotate(r1,location))
    r2 = o3d.geometry.get_rotation_matrix_from_xyz([0,0,np.pi/180])
    if not ROI:
        return [test_lines]
    else:
        # results = [test_lines,line1,line2]
        results = [line1,line2]
    for i in range(90):    
        line.rotate(r2,location)
        line3 = o3d.geometry.LineSet()
        line3.lines = o3d.utility.Vector2iVector([[0, 1]])
        line3.points = o3d.utility.Vector3dVector(np.array([np.asarray(line2.points)[1], np.asarray(line.points)[1]]))
        line3.colors = o3d.utility.Vector3dVector([color])
        results += [line3]
        line2 = copy.deepcopy(line)
    return results

def get_ego_bboxes(ego_label, location, rotation, calib, color=[1, 1, 1], image_location=None, fusion=None, ROI=None, fusion_location=None,fusion_rotation=None, ego_vehicle_data=None,index=None):
    image_location = np.array(image_location)
    
    ROI_rotation = rotation
    if image_location.any()!=None and not fusion:
        # image_location = np.array(image_location)
        if fusion_rotation is not None:
            rotation = fusion_rotation
        pass
    elif image_location.any()!=None and fusion:
        image_location = np.array(image_location) + fusion_location
        if fusion_rotation is not None:
            rotation = fusion_rotation
    else:
        image_location = location
        if fusion_rotation is not None:
            rotation = fusion_rotation
    # if image_location:
    #     location = np.array(image_location)# + location
    # print(color)
    # print(image_location)
    sensor_world_matrix = get_matrix(image_location, rotation)
    ego_ROI = o3d.geometry.TriangleMesh.create_cylinder(radius=110,height=0.1,resolution=100)
    
    # if image_location:
    #     ego_ROI.translate(image_location+[0,0,-10])
    #     ego_ROI.paint_uniform_color(color)
    # print(color)
    if color == [0.1,0.5,0.5]:
    # if color == [1]:
        ego_bboxes = []
    else:
        ego_vehicle = o3d.geometry.TriangleMesh.create_cone(radius=2,height=5)
        R = o3d.geometry.get_rotation_matrix_from_xyz([0,np.pi/2,0])
        ego_vehicle.rotate(R,center=[0,0,0])
        ego_rotation = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
        ego_vehicle.translate(np.array(ego_vehicle_data[:3]))
        ego_vehicle.rotate(ego_rotation,center=np.array(ego_vehicle_data[:3]))
        ego_vehicle.paint_uniform_color(color)
        ROI = True
        # ego_vehicle
        ego_bboxes = [ego_vehicle] + get_ego_ROI(location,ROI,ROI_rotation,image_location,color)
        # return get_ego_ROI(location,ROI,ROI_rotation,image_location,[0.1,0.1,0.1]),''
    if ego_vehicle_data is not None:
        bbox_center = ego_vehicle_data[:3]
        bbox_extend = ego_vehicle_data[3:6]
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([0,0,ego_vehicle_data[-1]]))
        bbox = o3d.geometry.OrientedBoundingBox(bbox_center, bbox_R, bbox_extend)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        lineset.paint_uniform_color(color)
        # ego_bboxes += [lineset]
        delta = ego_vehicle_data[-1]
        while delta > 1.57:
            delta -= np.pi
        while delta < -1.57:
            delta += np.pi
        data = [ego_vehicle_data[:-1]+[delta]+[10]+[index]]

    else:
        pass
        data = []
    # print(len(ego_label))
    for label in ego_label:
        if color != [0.6, 0.5, 0.5] and float(label[-1]) < -1:
            # print(label)
            continue
        # print('??')
        # tmp = -float(label[14])+rotation[2]-np.pi/2
        # if color!= [0.9,0.1,0.1]:
        #     print(label[4:8])        
        bbox_center = np.array(list(map(float, label[11:14]))).reshape(
            1, 3) - np.array([0, float(label[8])/2, 0])
        # print(bbox_center)
        bbox_center = calib.rect_to_lidar(bbox_center).flatten()
        bbox_center = np.append(bbox_center, 1).reshape(4, 1)
        bbox_center = np.array(np.dot(sensor_world_matrix, bbox_center).tolist()[:3])
        # if color!= [0.9,0.1,0.1]:
        # print(bbox_center)
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz([0,0,-float(label[14])+rotation[2]-np.pi/2])
        bbox_extend = np.array(list(map(float, label[8:11]))[::-1])
        bbox = o3d.geometry.OrientedBoundingBox(
            bbox_center, bbox_R, bbox_extend)
        # ego_bboxes.append(bbox)
        # print(bbox)
        delta = -float(label[14])+rotation[2]-np.pi/2
        while delta > np.pi/2:
            delta -= np.pi
        while delta < -np.pi/2:
            delta += np.pi
        data.append(bbox_center.flatten().tolist()+bbox_extend.tolist()+[delta]+[float(label[-1])]+[index])
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.1,cone_radius=0.3,cylinder_height=bbox_extend[0]*2/3,cone_height=bbox_extend[0]/3)
        R = o3d.geometry.get_rotation_matrix_from_xyz([0,np.pi/2,0])
        arrow.rotate(R,center=[0,0,0])
        arrow.paint_uniform_color(color)
        arrow.translate(bbox_center)
        arrow.rotate(bbox_R, bbox_center)
        # ego_bboxes.append(arrow)
        # print(arrow)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
        # lineset.rotate(R,location)
        # if fusion:
        #     # print('tset')
        #     lineset.paint_uniform_color([0.9,0.9,0.9])
        # else:
        #     # print(color)
        #     lineset.paint_uniform_color(color)
        lineset.paint_uniform_color(color)
        ego_bboxes.append(lineset)
    return ego_bboxes,data


def get_global_bboxes(global_labels):
    global_bboxes = []
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0, 0, np.pi])
    for _label in global_labels:
        if 'sensor' in _label[0]:
            continue
        bbox_center = np.array(
            list(map(float, _label[2:5]))) + np.array([0, 0, float(_label[10])])
        bbox_center[1] *= -1
        # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(list(map(np.radians,map(float, _label[5:8]))))
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz(
            np.array([0, 0, -np.radians(float(_label[7]))]))
        bbox_extend = np.array([float(num)*2 for num in _label[8:11]])
        global_bbox = o3d.geometry.OrientedBoundingBox(
            bbox_center, bbox_R, bbox_extend)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(
            global_bbox)
        # lineset.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
        # lineset.paint_uniform_color([0.1,0.1,0.1])
        lineset.paint_uniform_color([1,1,1])
        # lineset.rotate(R,center=[0,0,0])
        global_bboxes.append(lineset)
        # print('-----------')
    return global_bboxes

from utils.visualization_labels import get_fov_flag#,get_pcd
def get_global_pcd(pointcloud, color=[0.5, 0.5, 0.5], sensor_center=np.array([0, 0, 0]), sensor_rotation=np.array([0, 0, 0]), location=np.array([0, 0, 0]), calib=None, fusion_location=np.array([0, 0, 0])):
    # print(calib)
    calib = False
    if not calib:
        pcd = o3d.geometry.PointCloud()
        pointcloud = pointcloud[:, :3] + location# + sensor_center
        flag = pointcloud[:,2] > 1
        pointcloud = pointcloud[flag]
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        R = o3d.geometry.get_rotation_matrix_from_xyz(sensor_rotation)
        pcd.rotate(R,location)
        # pcd.paint_uniform_color([1.,1.,1.])
        pcd.paint_uniform_color([0, 0, 0])
        return [pcd]#+[test_lines]
    else:
        # print(location,'---',fusion_location)
        img_shape = [1242,375]
        pcd = o3d.geometry.PointCloud()
        pointcloud = pointcloud[:, :3]# + fusion_location
        pts_rect = calib.lidar_to_rect(pointcloud[:, 0:3])
        fov_flag = get_fov_flag(pts_rect, img_shape, calib)
        points = pointcloud[fov_flag][:,:3]
        # print(points.shape)
        points += fusion_location + location
        pcd.points = o3d.utility.Vector3dVector(points)
        R = o3d.geometry.get_rotation_matrix_from_xyz(sensor_rotation)
        pcd.rotate(R,fusion_location + location)
        pcd.paint_uniform_color(color)
        # pr
        return [pcd]

def get_pcd(pointcloud, color=[0.5, 0.5, 0.5], sensor_center=np.array([0, 0, 0]), sensor_rotation=np.array([0, 0, 0])):
    pcd = o3d.geometry.PointCloud()
    pointcloud = pointcloud[:, :3] + sensor_center
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.paint_uniform_color(color)
    return [pcd]


def visualization(labels, pointcloud):
    pcd = get_pcd(pointcloud)
    bboxes = get_bboxes(labels)
    o3d.visualization.draw_geometries(
        bboxes+pcd, width=960, height=640, point_show_normal=True)


def get_sensor_transform(vehicle_id, labels):
    for label in labels:
        if vehicle_id == label[-1] and 'lidar' in label[0]:
            sensor_center = np.array(list(map(float, label[2:5])))
            # sensor_center[1] *= -1
            sensor_rotation = np.array([0, 0, np.radians(float(label[5]))])
            return sensor_center, sensor_rotation


def vis_init():
    edges = [[[960,640,0],[960,-640,0]],
            [[-960,-640,0],[-960,640,0]],
            [[960,640,0],[-960,640,0]],
            [[960,-640,0],[-960,-640,0]]
            ]
    edges_lines= []
    for edge in edges:
        test_lines = o3d.geometry.LineSet()
        test_lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        test_lines.points = o3d.utility.Vector3dVector(np.array(edge)/2)
        test_lines.colors = o3d.utility.Vector3dVector([np.array([1,0,0])])
        edges_lines.append(test_lines)
    return edges_lines

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(1,0,77,-20)
    return False

def custom_draw_geometry(vis, geometry_list, map_file=None, recording=False,param_file='test.json',test=0):
    vis.clear_geometries()
    paper = True
    for p in geometry_list:
        R = o3d.geometry.get_rotation_matrix_from_xyz([0,0,np.pi*2*test/360])
        # p.rotate(R,[0,0,0])
    # from testo3d import get_strong
    # if paper:
    #     geometry_list = get_strong(geometry_list)

    # R = o3d.geometry.get_rotation_matrix_from_xyz([0,0,np.pi*2/36])
    R = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2,0,0])
    for pcd in geometry_list:

        # pcd.rotate(R,[0,0,0])
        vis.add_geometry(pcd)
    param = o3d.io.read_pinhole_camera_parameters(param_file)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    ctr = vis.get_view_control()
    # ctr.set_zoom(0.4)
    # ctr.set_up((0, -1, 0))
    # ctr.set_front((1, 0, 0))
    ctr.convert_from_pinhole_camera_parameters(param)
    # vis.register_animation_callback(rotate_view)
    vis.run()
    # time.sleep(5)
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters('test.json', param)
    if recording:
        vis.capture_screen_image(map_file,True)
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # if test == 1:
        o3d.io.write_pinhole_camera_parameters(param_file, param)


def get_ego_file(test_path, frame_id, task='pretrain'):
    ego_calib_file = test_path + '/calib00/' + frame_id
    ego_label_file = test_path + '/label00/' + frame_id
    ego_data_file = test_path + '/' + task + '/' + frame_id
    ego_pointcloud_file = test_path + '/velodyne/' + frame_id[:-3] +'bin'
    return ego_calib_file, ego_label_file, ego_data_file, ego_pointcloud_file


def get_ego_file_(data_path, ego_vehicle_label, global_label_file):
    ego_point_cloud_file = data_path + ego_vehicle_label[0] + '_' + str(ego_vehicle_label[1]) + \
        '/velodyne/' + global_label_file[:-3] + 'bin'
    ego_label_file = data_path + ego_vehicle_label[0] + '_' + str(ego_vehicle_label[1]) + \
        '/label/' + global_label_file[:-3] + 'txt'
    return ego_point_cloud_file, ego_label_file


def get_ego_vehicle(vehicle_id, global_label):
    for vehicle in global_label:
        if vehicle_id == vehicle[1]:
            # print(vehicle)
            location = list(map(float, vehicle[2:5]))
            # rotation = np.array(list(map(float, vehicle[5:8])))
            rotation = [0,0,np.radians(float(vehicle[7]))]
            location[1] *= -1
            rotation[2] *= -1
            extend = np.array(list(map(float, vehicle[8:11])))*2
            location[2] += extend[2]/2
            return location + extend.tolist() + [rotation[2]]

def get_ego_location(vehicle_id, global_label):
    for vehicle in global_label:
        if vehicle_id == vehicle[-1] and 'lidar' in vehicle[0]:
            location = np.array(list(map(float, vehicle[2:5])))
            # rotation = np.array(list(map(float, vehicle[5:8])))
            rotation = np.array([0,0,np.radians(float(vehicle[7]))])
            location[1] *= -1
            rotation[2] *= -1
            ego_vehicle = get_ego_vehicle(vehicle_id, global_label)
            return location, rotation, ego_vehicle


def get_ego_data(ego_data_path):
    try:
        try:
            ego_data = np.loadtxt(ego_data_path, dtype='str',
                                delimiter=' ').reshape((-1, 16))
        except:
            ego_data = np.loadtxt(ego_data_path, dtype='str',
                           delimiter=' ').reshape((-1, 15))
    except OSError:
        ego_data = []
    return ego_data

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import copy 

def dynamic_weight(label_data,fusion_task):    
    number = len(label_data)
    if fusion_task == 'dynamic':
        #dynamic weight
        weight = 1 / (1 + np.exp(-label_data[:,7]))
        weight = (weight / sum(weight)).reshape(1,-1)
        fusion_item = np.dot(weight,label_data).reshape(9,)
        return fusion_item,number
    elif fusion_task == 'mean':
        #mean
        return np.mean(label_data,axis=0),number
    elif fusion_task == 'max':
        index = np.argmax(label_data[:,7],axis=0)
        return label_data[index].reshape(9,),number

def judge_delta(label_data,fusion_task):
    label_data = np.array(label_data)
    max_delta = np.max(label_data[:,6])
    min_delta = np.min(label_data[:,6])
    if max_delta - min_delta > 2:
        for data in label_data:
            if data[6] < 0:
                data[6] += 3.14
    # fusion_item = np.mean(label_data,axis=0)
    fusion_item,number = dynamic_weight(copy.deepcopy(label_data),fusion_task)
    # print(fusion_item)
    fusion_item[7] = max(np.array(label_data)[:,7])
    return fusion_item, label_data, number

def dbscan(data,vis=None,geometry_list=None,frame_id=None,map_file=None,times=1,fusion_task='dynamic'):
    # clustering
    data = np.array(data)
    # data_norm = np.concatenate((data[:,:6],np.cos(data[:,7]*2).reshape(-1,1)),axis=1)
    data_norm = data[:,:7]
    # print(data_norm.shape) 
    # exit()
    db = DBSCAN(eps=1.5, min_samples=1).fit(data_norm)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    results = {}
    for index,label in enumerate(labels):
        if label not in results.keys():
            results[label] = [data[index]]
        else:
            results[label].append(data[index])
    
    #non maximum suppression
    tmp_results = {}
    for _,label in results.items():
        fusion_item,label_data,_ = judge_delta(label,fusion_task)    
        tmp = list(map(str,fusion_item))
        tmp_result = ' '.join(tmp)
        tmp_results[tmp_result] = label_data

    # from my_utils import nms_3d as nms3d 
    # from eval import d3_box_overlap
    # from d3iou import box3d_iou as iou3d
    tmp_list = np.array([list(map(float,tmp.split(' ')))[:8] for tmp in tmp_results.keys()]).reshape(-1,8)
    tmp_bboxes = tmp_list[:,:7]
    tmp_scores = tmp_list[:,7].flatten()   
    sorted_inds = np.argsort(-tmp_scores)
    keep_inds = []
    # while len(sorted_inds)>0:
    #     if tmp_scores[sorted_inds[0]] < -1: 
    #         break
    #     keep_inds.append(sorted_inds[0])
    #     if len(sorted_inds) == 1:
    #         break
    #     keep_box = tmp_bboxes[sorted_inds[0], :]
    #     res_inds = sorted_inds[1:]
    #     res_boxes = tmp_bboxes[res_inds, :]
    #     se_inds = np.arange(len(res_inds))

    #     def mask_3diou(keep_box, res_boxes):
    #         ious = []
    #         for res_bbox in res_boxes:
    #             iou_3d,iou_2d = iou3d(keep_box,res_bbox)
    #             ious.append(iou_3d)
    #         # print(ious,len(ious))
    #         return np.array(ious)
    #     # print(tmp_scores[res_inds],len(tmp_scores[res_inds]))

    #     ious = mask_3diou(keep_box, res_boxes)  # (m, )
    #     delete_mask = ious > 0.1
    #     delete_se_inds = se_inds[delete_mask]
    #     sorted_inds = np.delete(res_inds, delete_se_inds)
    # tmp_list = tmp_list[keep_inds,:]
    # print(tmp_list.shape)
    # exit()
    #fusion
    distill_flag = []
    for index in range(1,times+1):
        fusion_list = []
        fusion_label_list = []
        for label,label_data in tmp_results.items():
            # print(results)
            if list(map(float,label.split(' ')))[:8] not in tmp_list:
                continue
            if len(label_data) > 1 and np.max(label_data,axis=0)[6] - np.min(label_data,axis=0)[6] > 1.57:
                pass
                # print(np.max(label_data,axis=0)[-3] - np.min(label_data,axis=0)[-3])
                # if np.min(label_data,axis=0)[6] < -0.5:
                #     print(label_data)
            fusion_item,label_data,number = judge_delta(label_data,fusion_task)
            if max(np.array(label_data)[:,7]) > -0.8:#-0.7:# and max(np.array(label_data)[:,-2]) > -0.6:# -0.5:#-0.1:
                fusion_item[7] = max(np.array(label_data)[:,7])
                fusion_item[8] = number
                fusion_label_list.append(fusion_item[:9])        
                
                R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,fusion_item[6]])
                bbox = o3d.geometry.OrientedBoundingBox(fusion_item[:3],R,fusion_item[3:6])
                lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
                lineset.paint_uniform_color(np.array([0.9, 0.1, 0.1]))    
                if index == times:
                    fusion_list.append(lineset)

                    for label in label_data:
                        test = np.array(label[:6]) - np.array(fusion_item[:6])
                        test = np.linalg.norm(test)
                        if test > 0.5:
                            distill_flag.append(label[8])
                else:
                    for label in label_data:
                        tmp_color = color[int(label[8])]
                        label = fusion_item + (times-index)*(label-fusion_item)/times
                        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,label[6]])
                        bbox = o3d.geometry.OrientedBoundingBox(label[:3],R,label[3:6])
                        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
                        lineset.paint_uniform_color(tmp_color)
                        fusion_list.append(lineset)
            else:
                pass
                # print(fusion_item[-2])
        if not os.path.exists(map_file):
            os.makedirs(map_file)
        fusion_path = map_file+'/'+frame_id[:-4] +'-'+ str(index)+'.png'
        # custom_draw_geometry(vis,geometry_list+fusion_list,param_file='fusion_test.json',map_file=fusion_path,recording=True)
    return fusion_list,fusion_label_list,distill_flag

def process_theta(alpha):
    while alpha > np.pi:
        alpha -= np.pi*2
    while alpha < -np.pi:
        alpha += np.pi*2
    return alpha

import utils.kitti_common as kitti
# from eval import get_official_eval_result, get_coco_eval_result
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

if __name__ == "__main__":
    visualization_o3d = True
    one_frame = False
    fusion_task = 'dynamic'
    if one_frame:
        tmp_frame_id = '3945'
        tmp_vehicle_id = '485'

    if visualization_o3d:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960*2, height=640*2)
        # vis.get_render_option().background_color = np.array([0, 0, 0])
        vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.get_render_option().show_coordinate_frame = False
        vis.get_render_option().point_size = 1
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    
    root_path = sys.argv[1] #dataset path
    if len(sys.argv)==4:
        img_list = sys.argv[2] #dataset list path 'txt'
        task = sys.argv[3] #fusion method (pretrain, federated, distill)
    else:
        img_list = None
    test_list_path = root_path
    vehicle_list = [v for v in os.listdir(test_list_path) if 'vehicle' in v]
    road_list = [r for r in os.listdir(test_list_path) if 'road' in r]
    test_list = vehicle_list + road_list 

    print('vehicle numbers:',len(test_list))
    global_gt_path = root_path + '/global_label'
    frame_id_list = os.listdir(global_gt_path)
    if img_list is not None:
        img_list_file = root_path + '/' + img_list + '.txt'
        frame_id_list = np.loadtxt(img_list_file, dtype='str', delimiter=' ')
    frame_id_list.sort()
    print('frame_id_list numbers:',len(frame_id_list))

    distill_id = {}
    for test in test_list:
        distill_id[test] = []
   
    for frame_id in frame_id_list:
        frame_id = frame_id + '.txt'
        # if '7986' not in frame_id:
        #     continue
        print(int(frame_id[:-4]))
        if not int(frame_id[:-4]) >= 0:
            continue

        if one_frame:
            if tmp_frame_id not in frame_id:
                continue
            else:
                tmp_frame_id = frame_id
        else:
            pass

        if visualization_o3d:
            geometry_list = []
        
        #global label
        frame_global_label_path = global_gt_path + '/' + frame_id
        frame_label = np.loadtxt(
            frame_global_label_path, dtype='str', delimiter=' ')
   
        global_bboxes = get_global_bboxes(frame_label)
        # geometry_list += global_bboxes

        cluster_data = []
        # print(test_list)
        # ccc
        for index,test_id in enumerate(test_list):
            # if one_frame:
            #     if tmp_vehicle_id not in test_id:
            #         continue
            #     else:
            #         tmp_vehicle_id = test_id
            # else:
            #     pass
            test_path = test_list_path + '/' + test_id
            ego_calib_file, ego_label_file, ego_data_file, ego_pointcloud_file = get_ego_file(
                test_path, frame_id, task)
            calib = Calibration(ego_calib_file)
            ego_location, ego_rotation, ego_vehicle = get_ego_location(
                test_id[-3:], frame_label)
            color_black = [0, 0, 0]

            if 'model3' not in test_id:
                ego_bboxes,ego_cluster_data = get_ego_bboxes(get_ego_data(ego_data_file), ego_location, ego_rotation, calib, color=color[1], ego_vehicle_data=ego_vehicle,index=index)           

            if 'model3' in test_id:
                ego_bboxes,ego_cluster_data = get_ego_bboxes(get_ego_data(ego_data_file), ego_location, ego_rotation, calib, color=color[2], ego_vehicle_data=ego_vehicle,index=index)           
         
            ego_gt_bboxes,__ = get_ego_bboxes(get_ego_data(ego_label_file), ego_location, ego_rotation, calib, color=color[0], ego_vehicle_data=ego_vehicle)
            

            cluster_data += ego_cluster_data
            if visualization_o3d:
                geometry_list += ego_gt_bboxes #+ ego_bboxes
                geometry_list += ego_bboxes #+ ego_bboxes

        fusion_path = test_list_path+'/'+fusion_task+'/'+task +'_fusion_map' #(debug) <task>_fusion_map
        fusion_global_path = test_list_path+'/'+fusion_task+'/'+task +'_fusion/global/' #(label) <task>_fusion/global
        
        if not os.path.exists(fusion_global_path):
            os.makedirs(fusion_global_path)
        fusion_list,global_vehicle_list,distill_flag = dbscan(cluster_data,vis,geometry_list,frame_id,fusion_path,fusion_task=fusion_task)
        # def process_global(labels):
        #     global_labels = []
        #     for label in labels:
        #         global_labels.append(['Car','0','0',str(alpha),x_min,y_min,x_max,y_max,
        #                                             str(other_vehicle[5]), str(other_vehicle[4]), str(other_vehicle[3]),
        #                                             str(coordinate_camera[0]), str(coordinate_camera[1]), str(coordinate_camera[2]), 
        #                                             str(delta),str(other_vehicle[-1])])
        # global_label = process_global(global_vehicle_list)
        np.savetxt(fusion_global_path + frame_id, np.array(global_vehicle_list)[:,:], fmt='%s', delimiter=' ')
        # print(np.array(global_vehicle_list)[:,-1])
        # if visualization_o3d:
        #     geometry_list += fusion_list

        for index,test_id in enumerate(test_list):
            if one_frame:
                if tmp_vehicle_id not in test_id:
                    continue
                else:
                    tmp_vehicle_id = test_id
            else:
                pass
            test_path = test_list_path + '/' + test_id
            ego_fusion_path = test_list_path + '/'+fusion_task+'/'+ task + '_fusion/' + test_id 
            if not os.path.exists(ego_fusion_path):
                os.makedirs(ego_fusion_path)
            ego_calib_file = test_path + '/calib00/' + frame_id
            calib = Calibration(ego_calib_file)
            ego_pointcloud_file = test_path + '/velodyne/' + frame_id[:-3] + 'bin'
            ego_point_cloud = np.fromfile(
                ego_pointcloud_file, dtype=np.dtype('f4'), count=-1).reshape([-1, 4])
            ego_location, ego_rotation,ego_extend = get_ego_location(
                test_id[-3:], frame_label)
            tmp_point_cloud = get_global_pcd(ego_point_cloud, color=[0.9,0.9,0.9], sensor_center=ego_location,sensor_rotation=ego_rotation,location=ego_location,calib=calib)
            # tmp_point_cloud = get_pcd(ego_point_cloud,sensor_center=ego_location,sensor_rotation=ego_rotation)
            geometry_list += tmp_point_cloud
            sensor_world_matrix = get_matrix(ego_location,ego_rotation)
            world_sensor_matrix = np.linalg.inv(sensor_world_matrix)

            label_list = []
            for other_vehicle in global_vehicle_list:
                number = other_vehicle[-1].copy()
                # print(number,'-------------')
                other_vehicle = other_vehicle[:-1]
                tmp_pcd = tmp_point_cloud.copy()
                R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,other_vehicle[6]])
                bbox = o3d.geometry.OrientedBoundingBox(other_vehicle[:3],R,other_vehicle[3:6]+0.2)
                tmp_points = tmp_pcd[0].crop(bbox)
                # print(len(tmp_points.points))
                points = np.array(bbox.get_box_points())
                points = np.concatenate((points,np.ones((8,1))),axis=1)

                points = np.dot(world_sensor_matrix, points.T).T[:,:3]
                points = np.asarray(points).reshape(8,3)

                img_points,img_depths = calib.lidar_to_img(points)
                other_vehicle_location = np.array(other_vehicle[:3])
                other_vehicle_location = np.append(other_vehicle_location, 1).reshape(4, 1)
                other_location = np.dot(world_sensor_matrix, other_vehicle_location).flatten().tolist()[0][:3]
                # print('nb',other_location)
                # other_location = np.array(other_vehicle_location) - np.array(ego_location)
                other_location[2] -= other_vehicle[5]/2
                if all(img_depths > 0) and len(tmp_points.points) > 5:
                    x_min,y_min = np.min(img_points,axis=0)
                    x_max,y_max = np.max(img_points,axis=0)
                    if x_max < 0 or y_max < 0 or x_min > 1242 or y_min > 375 or x_max-x_min > 1242 or y_max-y_min > 375:
                        continue
                    if x_min < 0:
                        x_min = 0
                    if x_max > 1242:
                        x_max = 1242
                    if y_min < 0:
                        y_min = 0
                    if y_max > 375:
                        y_max =375
                    coordinate_camera = calib.lidar_to_rect(np.array(other_location).reshape(1,3)).flatten()
                    delta =  - other_vehicle[6] + ego_rotation[2] - np.pi/2
                    delta = process_theta(delta)
                    alpha = - delta - math.atan(other_location[0]/other_location[2]) - np.pi
                    alpha = process_theta(alpha)

                    if len(tmp_points.points) > 5:
                        size = 0
                        if len(tmp_points.points) + other_location[0] < 250:
                            size = 1
                        if len(tmp_points.points) + other_location[0] < 125:
                            size = 2
                    label_list.append(['Car',
                                            str(int(number)),str(size),str(alpha),x_min,y_min,x_max,y_max,
                                            str(other_vehicle[5]), str(other_vehicle[4]), str(other_vehicle[3]),
                                            str(coordinate_camera[0]), str(coordinate_camera[1]), str(coordinate_camera[2]), 
                                            str(delta),str(other_vehicle[-1])])
            if len(label_list) == 0:
                label_list.append(['DontCare','-1','-1','-10','522.25','202.35','547.77','219.71','-1','-1','-1','-1000','-1000','-1000','-10','-10'])
            np.savetxt(ego_fusion_path + '/' + frame_id, np.array(label_list), fmt='%s', delimiter=' ')
            # custom_draw_geometry(vis, geometry_list,param='test_fusion.json')

        fusion_path = test_list_path + '/dynamic/label00_fusion_map'+'/'+frame_id[:-4] +'-'+ str(index)+'.png'
        custom_draw_geometry(vis, geometry_list, param_file='fusion_test.json',map_file=fusion_path, recording=True)   

        distill_flag = [test_list[t] for t in list(map(int,set(distill_flag)))]
        for tmp in distill_flag:
            distill_id[tmp].append(frame_id[:-4])
        if one_frame and visualization_o3d:
            custom_draw_geometry(vis, geometry_list)
    for tmp,value in distill_id.items():
        tmp_path = test_list_path + '/'+fusion_task+'/'+ task + '_fusion/' + tmp
        np.savetxt(tmp_path + '_distill_list.txt', np.array(value), fmt='%s', delimiter=' ')
    exit()
