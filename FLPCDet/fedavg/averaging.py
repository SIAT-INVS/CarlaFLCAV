#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def average_weights(w):
    counter = 0
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            try:
                w_avg[k] += w[i][k]
            except Exception as e:
                w_avg[k] = 2*w[0][k]
                counter += 1
                pass
            
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg
