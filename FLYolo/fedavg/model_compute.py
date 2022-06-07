#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import random
from torch import nn


def model_sub(w1, w2):
    counter = 0
    w_diff = copy.deepcopy(w1)
    
    for k in w1.keys():
        for i in range(1, len(w1)):
            try:
                w_diff[k] = torch.sub(w1[k], w2[k])
            except Exception as e: 
                # not dtype
                counter += 1
                pass

    return w_diff

def model_add(w1, w2):
    counter = 0
    w_diff = copy.deepcopy(w1)
    
    for k in w1.keys():
        for i in range(1, len(w1)):
            try:
                w_diff[k] = torch.add(w1[k], w2[k])
            except Exception as e: 
                # not dtype
                counter += 1
                pass

    return w_diff
