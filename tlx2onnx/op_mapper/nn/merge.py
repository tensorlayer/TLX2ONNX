#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper
from ..op_mapper import OpMapper
from ...common import make_node
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np

@OpMapper(['Concat'])
class Concat():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        # get inputs outputs
        in_name = node['in_nodes_name']
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        layer = node['node'].layer
        concat_dim = layer.concat_dim
        # make concat node
        out_v = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
        onnx_value.append(out_v)
        out_node, _ = make_node('Concat', inputs=[s for s in in_name], outputs=node['out_nodes_name'], axis=concat_dim)
        onnx_node.append(out_node)
        return onnx_node, onnx_value, onnx_init


@OpMapper(['Elementwise'])
class Elementwise():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        # get inputs outputs
        in_name = node['in_nodes_name']
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        layer = node['node'].layer
        combine_fn_name = cls.fn_dict(str(layer.combine_fn.__name__))
        print(combine_fn_name)
        # make combine_fn node
        out_v = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
        onnx_value.append(out_v)

        out = in_name[0]
        for i in np.arange(1, len(in_name)):
            if i == len(in_name) - 1:
                out_node, out = make_node(combine_fn_name, inputs=[out, in_name[i]], outputs=[out_name])
                onnx_node.append(out_node)
            else:
                out_node, out = make_node(combine_fn_name, inputs=[out, in_name[i]], outputs=[out_name + str(i)])
                onnx_node.append(out_node)
        return onnx_node, onnx_value, onnx_init

    @staticmethod
    def fn_dict(fn):
        # More operator operations can be added from here.
        _dict = {
            'matmul': 'MatMul',
            'add': 'Add',
        }
        return _dict[fn]