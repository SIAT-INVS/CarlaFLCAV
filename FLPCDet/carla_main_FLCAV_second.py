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


CLOUD_ITER_TOTAL = 2
EDGE_ITER_TOTAL = 2

t1 = time.time()
CLOUD_ITER = 0
edge_list = ['town05', 'town03']

while CLOUD_ITER < CLOUD_ITER_TOTAL:
    print('===================== Cloud Federated Learning %d ====================='%CLOUD_ITER )

    # load cloud model to the edge
    if CLOUD_ITER == 0: filename = './fedmodel/cloud/pretrain.pth'
    if CLOUD_ITER > 0: filename = './fedmodel/cloud/global.pth'

    model_dict0 = torch.load(filename)
    params0 = model_dict0['model_state']
    
    ckpt = {'model_state':params0,
            'optimizer_state': model_dict0['optimizer_state'],
            'it': model_dict0['it']}

    for e in edge_list:
        foldername = './fedmodel/' + e
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        torch.save(ckpt, foldername + '/pretrain.pth')
    del ckpt

    for e in edge_list:
        EDGE_ITER = 0
        while EDGE_ITER < EDGE_ITER_TOTAL:
            print('===================== EDGE FEDERATED LEARNING =====================:' + e)
            sp.run(['./edgeFL.py', '%d'%EDGE_ITER, e])
            EDGE_ITER += 1
    
    w_locals = []
    for e in edge_list:
        filename = './fedmodel/' + e + '/global.pth'
        model_dict0 = torch.load(filename)
        params0 = model_dict0['model_state']
        w_locals.append(params0)
        del(params0)


    # compute perfect global model
    params_avg = average_weights(w_locals)    

    # global model 
    ckpt = {'model_state':params_avg,
            'optimizer_state': model_dict0['optimizer_state'],
            'it': model_dict0['it']}

    filename = './fedmodel/cloud/global.pth'
    torch.save(ckpt, filename)

    del ckpt

    # finish one PFL iteration
    print('---------------------------------Cloud FL Iteration %r is completed.'%CLOUD_ITER)
    CLOUD_ITER += 1



