#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import random
from torch import nn


def average_power(w):
    counter = 0
    power = torch.zeros(1, device=torch.device('cuda'))
    temp1 = torch.zeros(1)
    temp2 = torch.zeros(1)
    uuu = 0

    for k in w[0].keys():
        for i in range(0, len(w)):
            try:
                # dtype tensor
                temp1 = torch.norm(w[i][k]).pow(2)
                temp2 = torch.div(temp1, w[i][k].numel())
                power = torch.add(power, temp2)
                uuu += w[i][k].numel()
            except Exception as e: 
                # not dtype
                counter += 1
                pass


    power_avg = power / len(w) / len(w[0])    
    
    return power_avg
