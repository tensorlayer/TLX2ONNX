#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper
from ..op_mapper import OpMapper
from ...common import make_node
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE

@OpMapper(['GaussianNoise'])
class GaussianNoise():
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
        mean = layer.mean
        scale = layer.stddev
        seed = layer.seed
        # make random normal node
        r_out = helper.make_tensor_value_info(out_name + '_r', dtype, shape=out_shape)
        onnx_value.append(r_out)
        r_node, out = make_node('RandomNormal', inputs='', outputs=[out_name + 'r'],
                                  dtype=dtype, mean=mean, scale=scale, seed=seed, shape=out_shape)
        onnx_node.append(r_node)

        a_out = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
        onnx_value.append(a_out)
        a_node, out = make_node('Add', inputs=[in_name, out], outputs=[out_name])
        onnx_node.append(a_node)
        return onnx_node, onnx_value, onnx_init