#!/usr/bin/env python3
import os
import subprocess as sp

sp.run(['bash', '-c', 'python carla_pretrain_yolo.py pretrain'] ) # pretrain
sp.run(['bash', '-c', 'python carla_main_FLCAV_yolo.py']) # federated learning
    