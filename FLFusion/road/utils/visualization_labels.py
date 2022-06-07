import os,sys,random,time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
# from text_3d import text_3d
# print('Please input the legal record path and vehicle id.')
# print('record path: MMDDHHMM    vehicle_id: *')
from utils.calibration import Calibration
color = [
            [1.0,0.0,0.0],
            [0.0,0.0,1.0],
            [1.0,0.5,0.5],
            [0.5,1.0,0.5],
            [0.5,0.5,1.0],
            [1.0,0.5,0.0],
            [0.0,1.0,1.0],
            [1.0,0.0,1.0],
            [0.0,1.0,0.0],
        ]

location = [
                [-300,300,0],
                [-300,0,0],
                [-300,-300,0],
                [0,-300,0],
                [0,300,0],
                [300,300,0],
                [300,0,0],
                [300,-300,0]
        ]
def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    # print(lines)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

def get_bboxes(labels,calib,rotation=None):
    bboxes = []
    meshes = []
    for label in labels:
        # if float(label[-1]) < -1:
        #     continue
        if label[0] != 'Car':
            continue
        bbox_center = np.array(list(map(float,label[11:14]))).reshape(1,3) - np.array([0,float(label[8]),0]) / 2
        # print(bbox_center)
        bbox_center = calib.rect_to_lidar(bbox_center).flatten()
        # bbox_center -= np.array([0,0,float(label[8])]) / 2
        # print(bbox_center)
        # bbox_R = o3d.geometry.get_rotation_matrix_from_xyz([0,0,1])
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz  (np.array([0,0,-np.pi/2-float(label[14])]))
        # print(-np.pi/2-float(label[14]))
        bbox_extend = np.array(list(map(float,label[8:11]))[::-1]) 
        bbox = o3d.geometry.OrientedBoundingBox(bbox_center,bbox_R,bbox_extend)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        mesh.translate(bbox_center)
        mesh.rotate(bbox_R,bbox_center)
        meshes.append(mesh)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        print(label[2])
        color = [1., 1., 0.]
        if int(label[2]) == 1:
            color = [0., 1., 0.]
        if int(label[2]) == 2:
            color = [1., 0., 0.]
        import copy
        print(color)
        lineset.paint_uniform_color(np.array(color))
        bboxes.append(lineset)
    return bboxes,meshes

def get_ego_label(ego_label,rotation,location):
    ego_bboxes = []
    for label in ego_label:
        bbox_center = np.array(list(map(float,label[12:15]))) + location
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz  (np.array([0,0,float(label[3])]) + rotation)
        bbox_extend = np.array(list(map(float,label[8:11])))
        bbox = o3d.geometry.OrientedBoundingBox(bbox_center,bbox_R,bbox_extend)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
        lineset.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
        ego_bboxes.append(lineset)
    return ego_bboxes

def get_global_bboxes(global_labels):
    global_bboxes = []
    meshes = []
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz([0,0,np.pi])
    for _label in global_labels:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        bbox_center = np.array(list(map(float,_label[2:5]))) + np.array([0,0,float(_label[-1])])
        bbox_R = o3d.geometry.get_rotation_matrix_from_xyz  (np.array([0,0,np.radians(float(_label[5]))]))
        bbox_extend = np.array([float(num)*2 for num in _label[-4:-1]])
        global_bbox = o3d.geometry.OrientedBoundingBox(bbox_center,bbox_R,bbox_extend)
        mesh.rotate(bbox_R,bbox_center)
        meshes.append(mesh)
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(global_bbox)
        lineset.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
        # lineset.rotate(R,center=[0,0,0])
        global_bboxes.append(lineset)
    return global_bboxes,meshes
    
def get_global_pcd(pointcloud,color=[0.5, 0.5, 0.5],sensor_center=np.array([0,0,0]),sensor_rotation=np.array([0,0,0]),location=np.array([0,0,0])):
    pcd = o3d.geometry.PointCloud()
    pointcloud = pointcloud[:,:3] + location #+ sensor_center
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    # R = o3d.geometry.get_rotation_matrix_from_xyz(sensor_rotation)
    # pcd.rotate(R)
    pcd.paint_uniform_color(color)

    test_lines = o3d.geometry.LineSet()
    test_lines.lines = o3d.utility.Vector2iVector([[0,1]])
    test_lines.points = o3d.utility.Vector3dVector(np.array([sensor_center,location]))
    test_lines.colors = o3d.utility.Vector3dVector([color])

    return [pcd]+[test_lines]
    
def get_pcd(pointcloud,color=[0.5, 0.5, 0.5],sensor_center=np.array([0,0,0]),sensor_rotation=np.array([0,0,0])):
    pcd = o3d.geometry.PointCloud()
    pointcloud = pointcloud[:,:3] + sensor_center
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    # R = o3d.geometry.get_rotation_matrix_from_xyz([0,0,np.pi/2])
    # pcd.rotate(sensor_rotation,center=[0,0,0])
    # pcd.paint_uniform_color(color)
    pcd.paint_uniform_color(np.array([1,1,1]))
    return [pcd]
    
def get_2Dlabels(labels):
    all_2Dlabels = []
    for label in labels:

        # if float(label[-1]) < -1:
        #     continue
        tmp_2Dlabel = np.array(list(map(float,label[4:8]))+[int(label[2])])
        # tmp_2Dlabel.append(label[2])
        all_2Dlabels.append(tmp_2Dlabel)

    return all_2Dlabels

def draw_2Dlabels(label2d,img,pcd):
    fig = plt.figure(figsize=(12.42, 3.75))
    ax = fig.add_subplot(111)
    plt.imshow(img)
    for rect in label2d:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        color = 'y'
        if (rect[-1]) == 1:
            color = 'g'
        if (rect[-1]) == 2:
            color = 'r'
        rect = plt.Rectangle((rect[0],rect[1]),width,height,fill=False,color=color)
        ax.add_patch(rect)
    # plt.scatter(pcd[:,0],pcd[:,1],s=np.ones(pcd.shape[0])*1)
    plt.axis('off')
    # plt.show()
    
    plt.savefig('/home/zzj/Desktop/tmp.png')

def process_pcd(pcd,calib_info):
    process_data = np.dot(pcd, np.dot(calib_info['Tr_velo2cam'].T, calib_info['R0'].T))
    process_data = np.hstack((process_data, np.ones((process_data.shape[0], 1), dtype=np.float32)))
    process_data = np.dot(process_data, calib_info['P2'].T)
    pts_img = (process_data[:, 0:2].T / process_data[:, 2]).T  # (N, 2)
    pts_rect_depth = process_data[:, 2] - calib_info['P2'].T[3, 2]  # depth in rect camera coord
    return pts_img, pts_rect_depth

def get_fov_flag(pts_rect, img_shape, calib):
    '''
    Valid point should be in the image (and in the PC_AREA_SCOPE)
    :param pts_rect:
    :param img_shape:
    :return:
    '''
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    # print(pts_rect_depth)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[0])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[1])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    pts_valid_flag = np.logical_and(pts_valid_flag, pts_rect_depth <= 1000)
    return pts_valid_flag

def visualization(labels,pointcloud,img,test,calib_info,rotation=None):
    img_shape = [1242,375]
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    calib = Calibration(calib_info)
    pts_rect = calib.lidar_to_rect(pointcloud[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, img_shape, calib)
    points,_ = calib.lidar_to_img(pointcloud[fov_flag][:,:3])
    # print(points.shape)
    pcd = get_pcd(pointcloud[fov_flag][:,:3])
    bboxes,meshes = get_bboxes(labels,calib,rotation)
    # o3d.visualization.draw_geometries(bboxes+pcd+[]+[],width=960,height=640,point_show_normal=False)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1023, height=1023, left=5, top=5)
    vis.get_render_option().background_color = np.array([0., 0., 0.])
    vis.get_render_option().show_coordinate_frame = False
    vis.get_render_option().point_size = 2
    from testo3d import get_strong
    bboxes = get_strong(bboxes)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('1130.json')
    ctr.convert_from_pinhole_camera_parameters(param)
    # ctr.set_zoom(0.8)
    for t in bboxes+pcd:
        vis.add_geometry(t)
    vis.run()
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('1130.json', param)
    vis.capture_screen_image('/home/zzj/Desktop/test.png',True)
    bboxes = get_2Dlabels(labels)
    draw_2Dlabels(bboxes,img,points)
    exit()

def get_sensor_transform(vehicle_id, labels):
    for label in labels:
        if vehicle_id == label[-1] and 'lidar' in label[0]:
            sensor_center = np.array(list(map(float,label[2:5])))
            sensor_rotation = np.array([0,0,np.radians(float(label[5]))])
            return sensor_center, sensor_rotation

def custom_draw_geometry(vis,pcds,map_file,recording=False):
    vis.clear_geometries()
    for pcd in pcds:
        vis.add_geometry(pcd)
    # vis.update_renderer()
    vis.run()
    if recording:
        vis.capture_screen_image(map_file)

def get_ego_file(data_path,ego_vehicle_label,global_label_file):
    ego_point_cloud_file = data_path + ego_vehicle_label[0] + '_' + str(ego_vehicle_label[1]) + \
                                                            '/velodyne/' + global_label_file[:-3] + 'bin'
    ego_label_file = data_path + ego_vehicle_label[0] + '_' + str(ego_vehicle_label[1]) + \
                                                            '/label00/' + global_label_file[:-3] + 'txt'
    ego_camera_flie = data_path + ego_vehicle_label[0] + '_' + str(ego_vehicle_label[1]) + \
                                                            '/image00/' + global_label_file[:-3] + 'png'
    return ego_point_cloud_file, ego_label_file, ego_camera_flie

if __name__ == "__main__":
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080,height=1080)
    data_path = 'dataset'
    map_path = 'map'
    if len((sys.argv[1])) == 8:
        record_time = '_'.join(['2020',str(sys.argv[1][:4]),str(sys.argv[1][-4:])])
        data_path += '/record' + record_time + '/'
        map_path += '/record' + record_time + '/'
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    if len(sys.argv) == 2:
        vehicle_id = None
        pass
    elif len(sys.argv) == 3:
        vehicle_id = str(sys.argv[2])
    record_list = os.listdir(data_path)
    if vehicle_id:
        vis.destroy_window()
        ego_vehicle = None
        for record in record_list:
            if vehicle_id in record:
                ego_vehicle = record
                break
        data_path += ego_vehicle + '/'
        label_path = data_path + 'label00'
        raw_data_path = data_path + 'velodyne'
        image_path = data_path + 'image00'
        calib_path = data_path + 'calib00'
        label_files = os.listdir(label_path)

        def _read_imageset_file(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            return ["{:010d}.txt".format(int(line)) for line in lines]
        # label_files = _read_imageset_file()
        # label_files = os.listdir('/home/zzj/Disk/carla.0.9.8/dataset/record2020_0903_0142/vehicle.tesla.model3_595/pretrain')
        label_files.sort()
        for _label in label_files:#[100:-100:100]:
            if '7944' not in _label:
                continue
            print(_label)
            label_file = label_path + '/' + _label[:-3] + 'txt'
            # label_file = '/home/zzj/Disk/carla.0.9.8/dataset/record2020_0903_0142/vehicle.tesla.model3_595/label00/' + _label
            # label_file = '/home/zzj/Disk/carla.0.9.8/dataset/record2020_0903_0142/dynamic/pretrain_fusion/vehicle.tesla.model3_595' +'/'+_label
            # label_fed_file = label_path 
            raw_data_file = _label[:-3] + 'bin'
            raw_data_file = raw_data_path + '/' + raw_data_file
            image_data_file = image_path + '/' + _label[:-3] + 'png'
            calib_info_file = calib_path + '/' + _label[:-3] + 'txt'
            test = o3d.io.read_image(image_data_file)
            img = np.array(o3d.io.read_image(image_data_file))
            label = np.loadtxt(label_file, dtype='str',delimiter=' ').reshape([-1,16])
            pointcloud = np.fromfile(raw_data_file, dtype=np.float32, count=-1).reshape([-1,4])
            # calib_info = get_calib_from_file(calib_info_file)
            rotation = np.array([np.radians(-float(test)) for test in _label[5:8]]) #+ [0,0,np.pi/2]
            # rotation = np.zeros(3)
            # print(rotation)
            # rotation = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
            visualization(label,pointcloud,img,test,calib_info_file,rotation)
            # exit()
    else:
        global_label_path = data_path + 'global_label/'
        global_label_list = os.listdir(global_label_path)
        global_label_list.sort()
        for global_label_file in global_label_list:
            if '1756' not in global_label_file:
                continue
            global_label = np.loadtxt(global_label_path+global_label_file, dtype='str',delimiter=' ')
            global_bboxes = get_global_bboxes(global_label)
            points = []
            index = 0
            vis_global = False
            texts = []
            for ego_vehicle_label in global_label:
                if 'vehicle.tesla' in ego_vehicle_label[0]:
                    sensor_center, sensor_rotation = get_sensor_transform(ego_vehicle_label[1],global_label)
                    ego_point_cloud_file, ego_label_file, ego_label_image = get_ego_file(data_path,ego_vehicle_label,global_label_file)
                    ego_point_cloud = np.fromfile(ego_point_cloud_file, dtype=np.dtype('f4'), count=-1).reshape([-1,4])
                    ego_label = np.loadtxt(ego_label_file, dtype='str',delimiter=' ').reshape((-1,16))
                    if vis_global:
                        points += get_global_pcd(ego_point_cloud,color[index],sensor_center,sensor_rotation,location[index])
                        points += get_ego_label(ego_label,sensor_rotation,location[index])
                        # text_position = location[index] + np.array([0,-110,0])
                        # texts += text_3d('cars:%d'%len(ego_label) , pos=text_position, font_size=1600)
                    else:
                        points += get_pcd(ego_point_cloud,color[index],sensor_center,sensor_rotation)
                    index += 1
            # texts += text_3d('frame:%s'%global_label_file[:-4] , pos=[-50,450,0], font_size=1600)
            map_file = map_path + global_label_file[:-3] + 'png'
            custom_draw_geometry(vis,global_bboxes+points+texts,map_file,recording=True)
            # o3d.visualization.draw_geometries(global_bboxes+points,width=960,height=640,point_show_normal=True)

# python visualization_labels.py 07212103 178 时间和车ID 230行 换frameID
