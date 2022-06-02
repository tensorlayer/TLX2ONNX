#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorlayerx as tlx
from tlx2onnx.op_mapper.op_mapper import OpMapper

def transpose_shape(shape, perm):
    return np.transpose(np.ones(shape), perm).shape

def to_numpy(tensor):
    return tlx.convert_to_numpy(tensor)


def convert_tlx_relu(inputs, outputs, act = None):
    opsets = OpMapper.OPSETS['ReLU']
    map_func, kw= opsets[1]
    kw = {"inputs" : inputs,
          "outputs" : outputs}
    return map_func(node = None, **kw)


def convert_tlx_elu(inputs, outputs, act = None):
    opsets = OpMapper.OPSETS['ELU']
    map_func, kw = opsets[1]
    kw = {"inputs": inputs,
          "outputs": outputs,
          "alpha": act.alpha}
    return map_func(node=None, **kw)


def convert_tlx_tanh(inputs, outputs, act = None):
    opsets = OpMapper.OPSETS['Tanh']
    map_func, kw = opsets[1]
    kw = {"inputs": inputs,
          "outputs": outputs}
    return map_func(node=None, **kw)


def convert_tlx_sigmoid(inputs, outputs, act = None):
    opsets = OpMapper.OPSETS['Sigmoid']
    map_func, kw = opsets[1]
    kw = {"inputs": inputs,
          "outputs": outputs}
    return map_func(node=None, **kw)


def convert_tlx_lrelu(inputs, outputs, act = None):
    opsets = OpMapper.OPSETS['LeakyReLU']
    map_func, kw = opsets[1]
    kw = {"inputs": inputs,
          "outputs": outputs,
          "alpha": act.negative_slope}
    return map_func(node=None, **kw)


def convert_tlx_softplus(inputs, outputs, act = None):
    opsets = OpMapper.OPSETS['Softplus']
    map_func, kw = opsets[1]
    kw = {"inputs": inputs,
          "outputs": outputs}
    return map_func(node=None, **kw)


def convert_tlx_softmax(inputs, outputs, act = None):
    opsets = OpMapper.OPSETS['Softmax']
    map_func, kw = opsets[1]
    kw = {"inputs": inputs,
          "outputs": outputs}
    return map_func(node=None, **kw)


tlx_act_2_onnx = {
    "ReLU" :  convert_tlx_relu,
    "ELU" : convert_tlx_elu,
    "Tanh" : convert_tlx_tanh,
    "Sigmoid": convert_tlx_sigmoid,
    "LeakyReLU" : convert_tlx_lrelu,
    "Softplus" : convert_tlx_softplus,
    "Softmax": convert_tlx_softmax,
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