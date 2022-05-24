#! /usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import tensorlayerx as tlx

def transpose_shape(shape, perm):
    return np.transpose(np.ones(shape), perm).shape

def to_numpy(tensor):
    return tlx.convert_to_numpy(tensor)