#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node
import numpy as np


@OpMapper(["ExpandDims"])
class ExpandDims():
    # suppport v1-v13

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node, onnx_value, onnx_init = [], [], []
        x_name = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        axis = node['node'].layer.axis

        # Only Expand first dim
        shape = np.array([1] + x_shape).astype(np.int64)
        shape_value = numpy_helper.from_array(shape, name='shape')
        onnx_init.append(shape_value)
        e_value = helper.make_tensor_value_info(out_name + '_e', dtype, [1] + x_shape)
        onnx_value.append(e_value)
        e_node, out = make_node('Expand', inputs=[x_name, 'shape'], outputs=[out_name + '_e'])
        onnx_node.append(e_node)

        if axis == -1 or axis == (len(out_shape) - 1):
            r_shape = np.array(x_shape + [1]).astype(np.int64)
        else:
            r_shape = np.array(x_shape[0:axis] + [1] + x_shape[axis:]).astype(np.int64)
        r_shape_value = numpy_helper.from_array(r_shape, name='r_shape')
        onnx_init.append(r_shape_value)
        t_node, out = make_node('Reshape', inputs=[out, 'r_shape'], outputs=[out_name])
        onnx_node.append(t_node)
        return onnx_node, onnx_value, onnx_init


@OpMapper(["Tile"])
class Tile():
    # suppport v1-v13

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node, onnx_value, onnx_init = [], [], []
        x_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        multiples = np.array(node['node'].layer.multiples).astype(np.int64)
        multiples_value = numpy_helper.from_array(multiples, name='multiples')
        onnx_init.append(multiples_value)
        e_node, out = make_node('Tile', inputs=[x_name, 'multiples'], outputs=[out_name])
        value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
        onnx_node.append(e_node)
        onnx_value.append(value)
        return onnx_node, onnx_value, onnx_init
