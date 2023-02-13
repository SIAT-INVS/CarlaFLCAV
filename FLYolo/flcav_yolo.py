#!/usr/bin/env python3
import os
import subprocess as sp
import argparse
import sys
import torch
import copy
import time
from fedavg.averaging import average_weights,average_weights_person
from fedavg.model_compute import model_add, model_sub
import numpy as np
import math
from copy import deepcopy
from yolov5.utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from resource_allocation import resource_allocator


class FLCAV_YOLO:
    def __init__(self, args):
        self.pretrained_data_path = 'pretrain'
        self.federated_data_path = ['town05', 'town03']
        self.num_sample = 100
        self.CLOUD_ITER_TOTAL = 1
        self.EDGE_ITER_TOTAL = 1
        self.wireless_budget = args.wireless_budget
        self.wireline_budget = args.wireline_budget


    def pretrain(self, num_sample):
        data_path = self.pretrained_data_path
        s = num_sample
        vehicle_list = [v for v in os.listdir('raw_data/'+data_path) if 'vehicle' in v]
        print('vehicle numbers:',len(vehicle_list))
        print(vehicle_list)

        for v in vehicle_list:
            dataset = 'raw_data/' + data_path + '/' + v + '/yolo_coco_carla.yaml'
            savefolder = 'fedmodels/' + data_path + '/' + str(s)
            pretrain_model = 'yolov5s.pt'
            sp.run(['bash', '-c', 'python3 yolov5/train_local.py --img 640 --batch 8 --epochs 50 --data ' + dataset + ' --cfg yolov5/models/yolov5s.yaml --weights '+ pretrain_model + \
                ' --save ' + savefolder + ' --spsz '+str(s)])

        # save the model to the cloud
        filename = savefolder + '/weights/best.pt'
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


    def federated(self, num_edge_rounds, num_cloud_rounds):
        t1 = time.time()
        CLOUD_ITER = 0
        self.EDGE_ITER_TOTAL = num_edge_rounds
        self.CLOUD_ITER_TOTAL = num_cloud_rounds

        edge_list = self.federated_data_path 

        while CLOUD_ITER < self.CLOUD_ITER_TOTAL:
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
                while EDGE_ITER < self.EDGE_ITER_TOTAL:
                    print('===================== EDGE FEDERATED LEARNING =====================:' + e)
                    self.edge_federated(EDGE_ITER, self.EDGE_ITER_TOTAL, e)
                    EDGE_ITER += 1
            
            w_locals = []
            import sys
            sys.path.append('yolov5')
            for e in edge_list:
                filename = './fedmodels/' + e + '/weights/global.pt'
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


    def edge_federated(self, EDGE_ITER, EDGE_ITER_TOTAL, e):
        # load wireless distortion from data
        ITER = EDGE_ITER
        ITER_TOTAL = EDGE_ITER_TOTAL
        data_path = e

        print(ITER) # print current iteration
        print(data_path) # print dataset path

        vehicle_list = [v for v in os.listdir('raw_data/'+data_path) if 'vehicle' in v]
        print('vehicle numbers:',len(vehicle_list))
        print(vehicle_list)

        if ITER == 0:
            for v in vehicle_list:
                dataset = 'raw_data/' + data_path + '/' + v + '/yolo_coco_carla.yaml'
                savefolder = 'fedmodels/' + data_path + '/' + v
                pretrain_model = 'fedmodels/' + data_path + '/weights/pretrain.pt'
                sp.run(['bash', '-c', 'python3 yolov5/train_local.py --img 640 --batch 8 --epochs 2 --data ' + dataset \
                + ' --cfg yolov5/models/yolov5s.yaml --weights ' + pretrain_model + ' --save ' + savefolder \
                + ' --spsz '+str(1000)])

        if ITER > 0:
            for v in vehicle_list:
                dataset = 'raw_data/' + data_path + '/' + v + '/yolo_coco_carla.yaml'
                savefolder = 'fedmodels/' + data_path + '/' + v
                last_model = 'fedmodels/' + data_path + '/weights/global.pt'
                sp.run(['bash', '-c', 'python3 yolov5/train_local.py --img 640 --batch 8 --epochs 2 --data ' \
                + dataset + ' --cfg yolov5/models/yolov5s.yaml --weights ' + last_model + ' --save ' + savefolder \
                + ' --spsz '+str(1000)
                ])

        # save model to dictionary
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

        foldername = './fedmodels/' + data_path + '/weights'
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        filename = foldername + '/global.pt'
        torch.save(ckpt, filename)
        del ckpt

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
    args = argparser.parse_args()

    flcav_yolo = FLCAV_YOLO(args)
    allocator = resource_allocator.Resource_Allocator()
    cnn_opt_array, yolo_opt_array, second_opt_array = allocator.allocate(args.wireless_budget, args.wireline_budget)
    flcav_yolo.pretrain(int(yolo_opt_array[0]))
    flcav_yolo.federated(int(yolo_opt_array[1]), int(yolo_opt_array[2]))


if __name__ == "__main__":
    # execute only if run as a script
    main()


    