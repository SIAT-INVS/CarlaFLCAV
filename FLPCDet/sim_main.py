#!/usr/bin/env python3
import os
import subprocess as sp

sp.run(['bash', '-c', 'python carla_pretrain_second.py pretrain'] ) # pretrain
sp.run(['bash', '-c', 'python carla_main_FLCAV_second.py']) # federated learning
    