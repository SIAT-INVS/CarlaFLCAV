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
from fedavg.model_compute import model_add, model_sub
from fedavg.model_mask import add_noise
import numpy as np
import math
from copy import deepcopy
from yolov5.utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first

# load wireless distortion from data
ITER = int(argv[1])
ITER_TOTAL = int(argv[2])
data_path = argv[3]

print(ITER) # print current iteration
print(data_path) # print dataset path

vehicle_list = [v for v in os.listdir('raw_data/'+data_path) if 'vehicle' in v]
print('vehicle numbers:',len(vehicle_list))
print(vehicle_list)

if ITER == 0: # no pretrain model
    for v in vehicle_list:
        dataset = 'raw_data/' + data_path + '/' + v + '/yolo_coco_carla.yaml'
        savefolder = 'fedmodels/' + data_path + '/' + v
        pretrain_model = 'fedmodels/' + data_path + '/weights/pretrain.pt'
        sp.run(['bash', '-c', 'python3 yolov5/train_local.py --img 640 --batch 8 --epochs 2 --data ' + dataset \
        + ' --cfg yolov5/models/yolov5s.yaml --weights ' + pretrain_model + ' --save ' + savefolder \
        + ' --spsz '+str(1000)])

if ITER > 0: # no pretrain model
    for v in vehicle_list:
        dataset = 'raw_data/' + data_path + '/' + v + '/yolo_coco_carla.yaml'
        savefolder = 'fedmodels/' + data_path + '/' + v
        last_model = 'fedmodels/' + data_path + '/weights/global.pt'
        sp.run(['bash', '-c', 'python3 yolov5/train_local.py --img 640 --batch 8 --epochs 2 --data ' \
        + dataset + ' --cfg yolov5/models/yolov5s.yaml --weights ' + last_model + ' --save ' + savefolder \
        + ' --spsz '+str(1000)
        ])

# save model to dictionary
import sys
sys.path.append('yolov5')
w_locals = []
for v in vehicle_list:
    model_update = './fedmodels/' + data_path + '/' + v + '/weights/best.pt'
    model_dict = torch.load(model_update)
    params = model_dict['model'].state_dict()

    if ITER == 0:
        model_last = './fedmodels/' + data_path + '/weights/pretrain.pt'
        model_dict_last = torch.load(model_last)
        params_last = model_dict_last['model'].state_dict()

    if ITER >= 1:
        model_last = './fedmodels/' + data_path + '/weights/global.pt'
        model_dict_last = torch.load(model_last)
        params_last = model_dict_last['model'].state_dict()

    w_locals.append(model_sub(params, params_last))
    del(params)

params_avg = average_weights(w_locals)
global_params = model_add(params_avg, params_last)
del(params_last)

edgemodel = model_dict['model']
edgemodel.load_state_dict(global_params, strict=False)  # load

ckpt = {'epoch': model_dict['epoch'],
        'best_fitness': model_dict['best_fitness'],
        'model': deepcopy(de_parallel(edgemodel)).half(),
        'ema': model_dict['ema'],
        'updates': model_dict['updates'],
        'optimizer': model_dict['optimizer'],
        'wandb_id': model_dict['wandb_id']}

foldername = './fedmodels/' + data_path + '/edge/weights'
if not os.path.exists(foldername):
    os.makedirs(foldername)
filename = foldername + '/global.pt'
torch.save(ckpt, filename)
del ckpt

# finish one FL iteration
print('---------------------------------FL Iteration %r is completed.'%argv[1])