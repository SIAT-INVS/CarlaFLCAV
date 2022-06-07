#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def add_noise(w_avg, power_avg, MSE):

    counter = 0
    w_out = copy.deepcopy(w_avg)
    eta = torch.sqrt(MSE * power_avg)

    for k in w_out.keys():
        try:
            noise = torch.mul(torch.randn(w_avg[k].size(), device=torch.device('cuda')), eta)
            w_out[k] = w_avg[k] + noise
        except Exception as e:
            counter += 1
            pass
            
    return w_out
