#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper
from ..op_mapper import OpMapper
from ...common import make_node
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE

@OpMapper('Dropout')
class Dropout():
    # supports v1-v15

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        dropout_prob = str(node['node'].layer.p)
        dropout_mode = node['node'].layer.is_train
        ONNX_TYPE = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]

        if dropout_mode == False:
            y_v = helper.make_tensor_value_info(node['out_nodes_name'][0], ONNX_TYPE, shape=node['out_tensors'][0])
            onnx_value.append(y_v)
            o_node, _ = make_node('Identity', inputs=[x], outputs=node['out_nodes_name'])
            onnx_node.append(o_node)
        elif dropout_mode == True:
            y_v = helper.make_tensor_value_info(node['out_nodes_name'][0], ONNX_TYPE, shape=node['out_tensors'][0])
            onnx_value.append(y_v)
            o_node, _ = make_node('Dropout', inputs=[x, dropout_prob], outputs=node['out_nodes_name'])
            onnx_node.append(o_node)
        else:
            raise Exception("Unexpected situation happend")

        return onnx_node, onnx_value, onnx_init