#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE

@OpMapper(['Scale'])
class Scale():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        # get inputs outputs
        in_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        layer = node['node'].layer
        # get weights
        w_name = layer.name + '/weights'
        weights = numpy_helper.from_array(arr=to_numpy(layer.scale), name=w_name)
        onnx_init.append(weights)

        # make concat node
        out_v = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
        onnx_value.append(out_v)
        out_node, _ = make_node('Mul', inputs=[in_name, w_name], outputs=[out_name])
        onnx_node.append(out_node)
        return onnx_node, onnx_value, onnx_init