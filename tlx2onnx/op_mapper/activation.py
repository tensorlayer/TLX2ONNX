#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper
from .op_mapper import OpMapper
from ..common import make_node
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE

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
