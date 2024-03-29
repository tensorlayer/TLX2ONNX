#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorlayerx as tlx
from tlx2onnx.op_mapper.op_mapper import OpMapper
from onnx import helper, numpy_helper

def transpose_shape(shape, perm):
    return np.transpose(np.ones(shape), perm).shape

def to_numpy(tensor):
    return tlx.convert_to_numpy(tensor)

def convert_padding(padding, input_shape, output_shape, kernel_shape, strides, dilations, spatial, data_format):
    if dilations is None:
        dilations = [1] * spatial
    if isinstance(padding, str):
        if padding == "SAME":
            pads = [0] * (spatial * 2)
            if data_format == "channels_last":
                input_shape = make_shape_channels_first(input_shape)
                output_shape = make_shape_channels_first(output_shape)

            if any(input_shape[i + 2] == -1 or output_shape[i + 2] == -1 for i in range(spatial)):

                auto_pad = "SAME_UPPER"

                return  auto_pad

            for i in range(spatial):
                pad = (
                    (output_shape[i + 2] - 1) * strides[i]
                    + dilations[i] * (kernel_shape[i] - 1) + 1
                    - input_shape[i + 2]
                )
                pad = max(pad, 0)
                pads[i] = pad // 2
                pads[i + spatial] = pad - pad // 2

            return pads

        elif padding == "VALID":
            auto_pad = "VALID"
            return auto_pad
    elif isinstance(padding, int):
        pads = [padding] * spatial * 2
        return pads
    elif isinstance(padding, tuple):
        return list(padding) * 2

def convert_w(w, data_format, spatial, w_name):
    w = tlx.convert_to_numpy(w)
    if tlx.BACKEND == 'tensorflow':
        if spatial == 2:
            w = np.transpose(w, axes=[3, 2, 0, 1])
        elif spatial == 1:
            w = np.transpose(w, axes=[2, 1, 0])
        elif spatial == 3:
            w = np.transpose(w, axes=[4, 3, 0, 1, 2])
        return numpy_helper.from_array(w, name=w_name)
    elif tlx.BACKEND == 'mindspore':
        if spatial == 2 and data_format == 'channels_last':
            w = np.transpose(w, axes=[3, 0, 1, 2])
            return numpy_helper.from_array(w, name=w_name)
    return numpy_helper.from_array(w, name=w_name)

def convert_b(b, b_name):
    b = tlx.convert_to_numpy(b)
    return numpy_helper.from_array(b, name=b_name)

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

def squeeze_axes(spatial):
    return list(range(2, spatial+2))