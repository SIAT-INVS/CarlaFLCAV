#!/usr/bin/env python3
import os
import subprocess as sp
import argparse
import sys
import torch
import copy
import time
from fedavg.averaging import average_weights
from fedavg.averaging_power import average_power
from fedavg.params_mask import add_randn_noise
import numpy as np
import math
from copy import deepcopy
from resource_allocation import resource_allocator


class FLCAV_SECOND:
    def __init__(self, args):
        self.pretrained_data_path = 'pretrain'
        self.federated_data_path = ['town03', 'town05']
        self.num_sample = 5
        self.CLOUD_ITER_TOTAL = 5
        self.EDGE_ITER_TOTAL = 1
        self.wireless_budget = args.wireless_budget
        self.wireline_budget = args.wireline_budget
        self.batch_size = args.batch_size
        self.epochs = args.epochs


    def pretrain(self, num_sample):
        data_path = self.pretrained_data_path
        s = num_sample

        vehicle_list = [v for v in os.listdir('tools/cfgs/kitti_models/'+data_path) if 'vehicle' in v]
        print('vehicle numbers:',len(vehicle_list))
        print(vehicle_list)

        for v in vehicle_list:
            cfg_yaml = 'cfgs/kitti_models/'+data_path + '/' + v 
            sp.run(['bash', '-c', 'cd tools; python pretrain.py --cfg_file {} --batch_size {} --epochs {} --spsz {}'.format(cfg_yaml, self.batch_size, self.epochs, s)]) # local model updates at vehicle v
            # sp.run(['bash', '-c', 'cd tools; python pretrain.py --cfg_file ' + cfg_yaml + ' --batch_size 1 --epochs 20 --spsz '+str(s)]) # local model updates at vehicle v 
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


    def federated(self, num_edge_rounds, num_cloud_rounds):

        t1 = time.time()
        CLOUD_ITER = 0
        self.EDGE_ITER_TOTAL = num_edge_rounds
        self.CLOUD_ITER_TOTAL = num_cloud_rounds

        edge_list = self.federated_data_path 

        while CLOUD_ITER < self.CLOUD_ITER_TOTAL:
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
                while EDGE_ITER < self.EDGE_ITER_TOTAL:
                    print('===================== EDGE FEDERATED LEARNING =====================:' + e)
                    self.edge_federated(EDGE_ITER, self.EDGE_ITER_TOTAL, e)
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


    def edge_federated(self, EDGE_ITER, EDGE_ITER_TOTAL, e):

        # load wireless distortion from data
        ITER = EDGE_ITER
        ITER_TOTAL = EDGE_ITER_TOTAL
        data_path = e

        print(ITER) # print current iteration
        print(data_path) # print dataset path

        # vehicle_list = [v for v in os.listdir('data/'+data_path) if 'vehicle' in v]
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
        print('---------------------------------FL Iteration %r is completed.'%ITER)



def main():
    argparser = argparse.ArgumentParser(description=__doc__)   
    argparser.add_argument(
        '-l', '--wireline_budget',
        metavar='L',
        default=4096,
        type=int,
        help='Wireline resource constraint in MB')

    argparser.add_argument(
        '-w', '--wireless_budget',
        metavar='W',
        default=4096,
        type=int,
        help='Wireless resource constraint in MB')
    
    argparser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='Batch size for training')
    
    argparser.add_argument(
        '--epochs',
        default=1,
        type=int,
        help='Number of epochs for training')

    args = argparser.parse_args()

    flcav_second = FLCAV_SECOND(args)
    allocator = resource_allocator.Resource_Allocator()
    cnn_opt_array, yolo_opt_array, second_opt_array = allocator.allocate(args.wireless_budget, args.wireline_budget)

    flcav_second.pretrain(int(second_opt_array[0]))
    flcav_second.federated(int(second_opt_array[1]), int(second_opt_array[2]))


if __name__ == "__main__":
    # execute only if run as a script
    main()


    