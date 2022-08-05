#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto
from ..op_mapper import OpMapper
from ...common import make_node, transpose_shape
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE

@OpMapper('Add')
class Add():
    # supports v7-v12

    @classmethod
    def version_7(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        x_name = node['in_nodes_name'][0]
        y_name = node['in_nodes_name'][1]
        out_name = node['out_nodes_name'][0]
        # x_shape = node['in_tensors'][0]
        # y_shape = node['in_tensors'][1]
        # out_shape = node['out_tensors'][0]

        op_type = 'Add'
        add_node, _ = make_node(op_type, inputs=[x_name, y_name], outputs=[out_name])
        onnx_node.append(add_node)
        return onnx_node, onnx_value, onnx_init

