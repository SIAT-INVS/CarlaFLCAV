#!/usr/bin/env python3
import os
import subprocess as sp
import torch
import pdb
import copy
import time
from sys import argv
from fedavg.averaging import average_weights
from fedavg.averaging_power import average_power
from fedavg.params_mask import add_randn_noise
import numpy as np


# load wireless distortion from data
ITER = int(argv[1])
data_path = argv[2]

print(ITER) # print current iteration
print(data_path) # print dataset path

vehicle_list = [v for v in os.listdir('tools/cfgs/kitti_models/'+data_path) if 'vehicle' in v]
print('vehicle numbers:',len(vehicle_list))
print(vehicle_list)


if ITER == 0: # no pretrain model
    for v in vehicle_list:
        cfg_yaml = 'cfgs/kitti_models/'+data_path + '/' + v 
        pretrain_model = '../fedmodel/' + data_path + '/pretrain.pth'
        sp.run(['bash', '-c', 'cd tools; python train_local.py --cfg_file ' + cfg_yaml \
        + ' --batch_size 1 --epochs 1 --pretrained_model '+pretrain_model]) # local model updates at vehicle v 


if ITER >=1: # load last round model
    for v in vehicle_list:
        cfg_yaml = 'cfgs/kitti_models/'+data_path + '/' + v 
        last_model = '../fedmodel/' + data_path + '/global.pth'
        sp.run(['bash', '-c', 'cd tools; python train_local.py --cfg_file ' + cfg_yaml \
        + ' --batch_size 1 --epochs 1 --pretrained_model ' + last_model]) # local model updates at vehicle v 

# append parameters into a list
w_locals = []
# # load model to dictionary
for v in vehicle_list:
    v_no = v.split('.')[0]
    local_model = './output/kitti_models/'+data_path+'/'+v_no+'/default/ckpt/local_model.pth'
    model_dict = torch.load(local_model); 
    model_params = model_dict['model_state']
    w_locals.append(model_params)
    # remove files
    del(model_params)
    os.remove(local_model)

# compute perfect global model
params_avg = average_weights(w_locals)

# pass perfect global parameters into noisy channels
# error = np.loadtxt('error2.txt')
# error = error * 0 # perfect 
power_avg = average_power(w_locals)
# MSE = torch.tensor(error[ITER][0], device=torch.device('cuda'))
MSE = torch.tensor(0, device=torch.device('cuda'))
params_avg = add_randn_noise(params_avg, power_avg, MSE)

# save noisy global model at the edge server
foldername = './fedmodel/'+data_path
filename = './fedmodel/'+data_path+'/global.pth'

if not os.path.exists(foldername):
            os.makedirs(foldername)

torch.save({'model_state':params_avg,
            'optimizer_state': model_dict['optimizer_state'],
            # 'epoch': model_dict0['epoch'],
            'it': model_dict['it']
            },filename)


# finish one FL iteration
print('---------------------------------FL Iteration %r is completed.'%argv[1])

