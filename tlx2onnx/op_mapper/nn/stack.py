#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE

@OpMapper(['Stack'])
class Stack():
    # supports v1-v12

    @classmethod
    def version_11(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        # get inputs outputs
        in_names = node['in_nodes_name']
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        layer = node['node'].layer
        axis = layer.axis

        # make concat node
        out_v = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
        onnx_value.append(out_v)
        seq_construct_node = helper.make_node('SequenceConstruct', [v for v in in_names], [layer.name + 'S'])
        onnx_node.append(seq_construct_node)
        out_node, _ = make_node('ConcatFromSequence', inputs=[layer.name + 'S'], outputs=[out_name], new_axis=1, axis=axis)
        onnx_node.append(out_node)
        return onnx_node, onnx_value, onnx_init


@OpMapper(['UnStack'])
class UnStack():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        # get inputs outputs
        in_name = node['in_nodes_name'][0]
        out_names = node['out_nodes_name']
        out_shape = node['out_tensors']
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        layer = node['node'].layer
        axis = layer.axis

        # make concat node
        out_node, _ = make_node('Split', inputs=[in_name], outputs=[v for v in out_names], axis=axis)
        onnx_node.append(out_node)
        return onnx_node, onnx_value, onnx_init