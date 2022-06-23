#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from ...common import tlx_act_2_onnx
import numpy as np


@OpMapper('Transpose')
class Transpose():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        # input, output
        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        y = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=out_shape)
        onnx_value.append(out_v)

        # attr
        perm = node['node'].layer.perm
        conjugate = node['node'].layer.conjugate

        if conjugate:
            raise NotImplementedError("parameter conjugate is not supported.")

        t_node, _ = make_node('Transpose', inputs=[x], outputs=[y], perm=perm)
        onnx_node.append(t_node)

        return onnx_node, onnx_value, onnx_init


@OpMapper('Reshape')
class Reshape():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        # input, output
        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        y = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=out_shape)
        onnx_value.append(out_v)

        # attr
        shape = np.array(node['node'].layer.shape, dtype=np.int64)
        shape_value = numpy_helper.from_array(shape, name='shape')
        onnx_init.append(shape_value)

        t_node, _ = make_node('Reshape', inputs=[x, 'shape'], outputs=[y])
        onnx_node.append(t_node)

        return onnx_node, onnx_value, onnx_init