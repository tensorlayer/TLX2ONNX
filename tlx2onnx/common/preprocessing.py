#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorlayerx as tlx
from tlx2onnx.op_mapper.op_mapper import OpMapper

def transpose_shape(shape, perm):
    return np.transpose(np.ones(shape), perm).shape

def to_numpy(tensor):
    return tlx.convert_to_numpy(tensor)


def convert_tlx_relu(inputs, outputs, name = None):
    opsets = OpMapper.OPSETS['ReLU']
    map_func, kw= opsets[1]
    kw = {"inputs" : inputs,
          "outputs" : outputs}
    return map_func(node = None, **kw)

def convert_tlx_elu(inputs, outputs, name = None):
    pass

def convert_tlx_tanh(inputs, outputs, name = None):
    pass

def convert_tlx_sigmoid(inputs, outputs, name = None):
    pass

def convert_tlx_lrelu(inputs, outputs, name = None):
    pass

def convert_tlx_softplus(inputs, outputs, name = None):
    pass

def convert_tlx_relu6(inputs, outputs, name = None):
    pass

tlx_act_2_onnx = {
    "ReLU" :  convert_tlx_relu,
    "Elu" : convert_tlx_elu,
    "Tanh" : convert_tlx_tanh,
    "Sigmoid": convert_tlx_sigmoid,
    "LeakyRelu" : convert_tlx_lrelu,
    "Softplus" : convert_tlx_softplus,
    "Relu6" : convert_tlx_relu6,
}

def make_shape_channels_first(shape):
    """Makes a (N, ..., C) shape into (N, C, ...)."""

    return shape[:1] + shape[-1:] + shape[1:-1]


def make_shape_channels_last(shape):
    """Makes a (N, C, ...) shape into (N, ..., C)."""

    return shape[:1] + shape[1:-1] + shape[1:2]

def get_channels_first_permutation(spatial):
    """Returns a permutation to make a (N, ..., C) array into (N, C, ...)."""

    return [0, spatial + 1] + list(range(1, spatial + 1))

def get_channels_last_permutation(spatial):
    """Returns a permutation to make a (N, C, ...) array into (N, ..., C)."""

    return [0] + list(range(2, spatial+2)) + [1]