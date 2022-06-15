#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper, TensorProto
from .op_mapper import OpMapper
from ..common import make_node
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np

__all__ = ['Relu', 'LeakyReLU', 'ELU', 'Tanh', 'Sigmoid', 'Softmax', 'Softplus', 'ReLU6']

@OpMapper(["ReLU"])
class Relu():

    @classmethod
    def version_1(cls, node, **kwargs):
        Op_name = 'Relu'
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None :
            # get in_node_name out_node_name out_tensor_shape
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            node, out = make_node(Op_name, inputs=[x_name], outputs=[out_name])
            value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_node.append(node)
            onnx_value.append(value)
            return onnx_node, onnx_value, onnx_init
        return make_node(Op_name, **kwargs)


@OpMapper(["LeakyReLU"])
class LeakyReLU():

    @classmethod
    def version_1(cls, node, **kwargs):
        return make_node('LeakyRelu', **kwargs)


@OpMapper(["ELU"])
class ELU():

    @classmethod
    def version_1(cls, node, **kwargs):
        return make_node('Elu', **kwargs)


@OpMapper(["Tanh"])
class Tanh():

    @classmethod
    def version_1(cls, node, **kwargs):
        return make_node('Tanh', **kwargs)


@OpMapper(["Sigmoid"])
class Sigmoid():

    @classmethod
    def version_1(cls, node, **kwargs):
        return make_node('Sigmoid', **kwargs)


@OpMapper(["Softmax"])
class Softmax():

    @classmethod
    def version_1(cls, node, **kwargs):
        return make_node('Softmax', **kwargs)


@OpMapper(["Softplus"])
class Softplus():

    @classmethod
    def version_1(cls, node, **kwargs):
        return make_node('Softplus', **kwargs)


@OpMapper(["ReLU6"])
class ReLU6():

    @classmethod
    def version_1(cls, node, **kwargs):

        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]

        relu_out = helper.make_tensor_value_info(node['in_nodes_name'][0] + 'r', dtype, shape=node['in_tensors'][0])
        onnx_value.append(relu_out)
        relu_node, out = make_node('Relu', [x], [node['in_nodes_name'][0] + 'r'])
        onnx_node.append(relu_node)

        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], dtype, shape=node['out_tensors'][0])
        onnx_value.append(out_v)

        max_v = np.array(6).astype(node['dtype'])
        max_value = numpy_helper.from_array(max_v, name='max_v')
        onnx_init.append(max_value)
        min_node, out = make_node('Clip', inputs=[out, "", 'max_v'], outputs=node['out_nodes_name'])
        onnx_node.append(min_node)

        return onnx_node, onnx_value, onnx_init