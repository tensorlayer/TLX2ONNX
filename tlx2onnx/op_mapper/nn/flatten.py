#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import STR_TYPE_TO_TENSOR_TYPE

@OpMapper('Flatten')
class Flatten():
    # supports v1-v15

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], STR_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=node['out_tensors'][0])
        onnx_value.append(out_v)
        out_node, _ = make_node('Flatten', inputs=[x], outputs=node['out_nodes_name'])
        onnx_node.append(out_node)

        return onnx_node, onnx_value, onnx_init