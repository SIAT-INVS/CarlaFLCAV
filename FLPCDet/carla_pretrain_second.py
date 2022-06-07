#!/usr/bin/env python3
import os
import subprocess as sp
import torch
import pdb
import copy
import time
from sys import argv
import numpy as np


data_path = argv[1] # dataset config path
print(data_path) # print dataset path

vehicle_list = [v for v in os.listdir('tools/cfgs/kitti_models/'+data_path) if 'vehicle' in v]
print('vehicle numbers:',len(vehicle_list))
print(vehicle_list)

sample_list = [5]

for s in sample_list:
    for v in vehicle_list:
        cfg_yaml = 'cfgs/kitti_models/'+data_path + '/' + v 
        sp.run(['bash', '-c', 'cd tools; python pretrain.py --cfg_file ' + cfg_yaml + ' --batch_size 1 --epochs 1 --spsz '+str(s)]) # local model updates at vehicle v 
    # finish one experiment
    print('---------------------------------Number of samples %r is completed.'%s)

foldername = './output//kitti_models/pretrain/vehicle_train/default/ckpt/'
filename = 'pretrained_model_' + str(s) +'.pth'

model_dict0 = torch.load(foldername+filename)
params0 = model_dict0['model_state']
ckpt = {'model_state':params0,
        'optimizer_state': model_dict0['optimizer_state'],
        'it': model_dict0['it']}

foldername = './fedmodel/cloud/'
if not os.path.exists(foldername):
    os.makedirs(foldername)
    torch.save(ckpt, foldername + 'pretrain.pth')
    del ckpt



# python carla_pretrain_second.py pretrain/train
