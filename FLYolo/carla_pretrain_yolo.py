#!/usr/bin/env python3
import os
import subprocess as sp
import torch
import pdb
import copy
import time
from sys import argv
from fedavg.averaging import average_weights
from fedavg.averaging_power import average_power,max_power
from fedavg.model_compute import model_add,model_sub
from fedavg.model_mask import add_noise
import numpy as np
import math
from copy import deepcopy
from yolov5.utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first


data_path = argv[1]
sample_list = [90]
vehicle_list = [v for v in os.listdir('raw_data/'+data_path) if 'vehicle' in v]
print('vehicle numbers:',len(vehicle_list))
print(vehicle_list)

for s in sample_list:
    for v in vehicle_list:
        dataset = 'raw_data/' + data_path + '/' + v + '/yolo_coco_carla.yaml'
        savefolder = 'fedmodels/' + data_path + '/' + str(s)
        pretrain_model = 'yolov5s.pt'
        sp.run(['bash', '-c', 'python3 yolov5/train_local.py --img 640 --batch 8 --epochs 50 --data ' + dataset + ' --cfg yolov5/models/yolov5s.yaml --weights '+ pretrain_model + \
         ' --save ' + savefolder + ' --spsz '+str(s)])

    # save the model to the cloud
    filename = savefolder + '/weights/best.pt'
    import sys
    sys.path.append('yolov5')
    model_dict0 = torch.load(filename)
    model0 = model_dict0['model']
    
    ckpt = {'epoch': model_dict0['epoch'],
        'best_fitness': model_dict0['best_fitness'],
        'model': deepcopy(de_parallel(model0)).half(),
        'ema': model_dict0['ema'],
        'updates': model_dict0['updates'],
        'optimizer': model_dict0['optimizer'],
        'wandb_id': model_dict0['wandb_id']}

    foldername = './fedmodels/cloud/weights'
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    filename = foldername + '/pretrain.pt'
    torch.save(ckpt, foldername + '/pretrain.pt')
    del ckpt

    # finish one iteration
    print('---------------------------------Number of samples %r is completed.'%s)



