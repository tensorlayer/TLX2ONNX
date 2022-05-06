#! /usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np

def transpose_shape(shape, perm):
    return np.transpose(np.ones(shape), perm).shape