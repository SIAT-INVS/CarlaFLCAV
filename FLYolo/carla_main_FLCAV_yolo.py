#!/usr/bin/env python3
import os
import subprocess as sp
import torch
import pdb
import copy
import time
from sys import argv
from fedavg.averaging import average_weights,average_weights_person
import numpy as np
import math
from copy import deepcopy
from yolov5.utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first


CLOUD_ITER_TOTAL = 1
EDGE_ITER_TOTAL = 1

t1 = time.time()
CLOUD_ITER = 0
edge_list = ['town05', 'town03']

while CLOUD_ITER < CLOUD_ITER_TOTAL:
    print('===================== Cloud FL %d ====================='%CLOUD_ITER )

    # load cloud model to the edge
    if CLOUD_ITER == 0: filename = './fedmodels/cloud/weights/pretrain.pt'
    if CLOUD_ITER > 0: filename = './fedmodels/cloud/weights/global.pt'

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

    for e in edge_list:
        foldername = './fedmodels/' + e + '/weights'
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        filename = './fedmodels/' + e + '/weights/pretrain.pt'
        torch.save(ckpt, foldername + '/pretrain.pt')
    del ckpt

    for e in edge_list:
        t1 = time.time()
        EDGE_ITER = 0
        while EDGE_ITER < EDGE_ITER_TOTAL:
            print('===================== EDGE FEDERATED LEARNING =====================:' + e)
            sp.run(['./carla_fed_yolo.py', '%d'%EDGE_ITER, '%d'%EDGE_ITER_TOTAL, e])
            EDGE_ITER += 1
    
    w_locals = []
    import sys
    sys.path.append('yolov5')
    for e in edge_list:
        filename = './fedmodels/' + e + '/edge/weights/global.pt'
        model_dict0 = torch.load(filename)
        model0 = model_dict0['model']
        params = model0.state_dict()
        w_locals.append(params)
        del(params)

    model_avg = model0
    model_dict_avg = model_dict0

    # compute perfect global model
    params_avg = average_weights(w_locals)    

    model_avg.load_state_dict(params_avg, strict=False)  # load
    del(params_avg)

    # global model 
    
    ckpt = {'epoch': model_dict_avg['epoch'],
        'best_fitness': model_dict_avg['best_fitness'],
        'model': deepcopy(de_parallel(model_avg)).half(),
        'ema': model_dict_avg['ema'],
        'updates': model_dict_avg['updates'],
        'optimizer': model_dict_avg['optimizer'],
        'wandb_id': model_dict_avg['wandb_id']}

    filename = './fedmodels/cloud/weights/global.pt'
    torch.save(ckpt, filename)

    del ckpt

    # finish one PFL iteration
    print('---------------------------------Cloud FL Iteration %r is completed.'%CLOUD_ITER)
    CLOUD_ITER += 1



