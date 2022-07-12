#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np

@OpMapper(['PadLayer'])
class PadLayer():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node, onnx_value, onnx_init = [], [], []
        # get inputs outputs
        in_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        layer = node['node'].layer
        # get attrs
        value = np.array(layer.constant_values).astype(node['dtype'])
        c_value = numpy_helper.from_array(value, name='value')
        onnx_init.append(c_value)
        # processing mode
        mode_dict = {"CONSTANT": 'constant', "REFLECT": 'reflect',"SYMMETRIC": 'edge'}
        mode = mode_dict[layer.mode]
        # processing padding. `pads` should be a 1D tensor of shape [2 * input_rank].
        # `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...],
        padding = layer.padding
        pads_temp = padding[0]
        for i in np.arange(1, len(padding)):
            pads_temp += padding[i]
        pads = []
        for i in range(len(pads_temp)//2):
            pads.append(pads_temp[2*i])
        for i in range(len(pads_temp) // 2):
            pads.append(pads_temp[i*2+1])
        pads = np.array(pads).astype(np.int64)
        p_value = numpy_helper.from_array(pads, name='pads')
        onnx_init.append(p_value)
        # make nodes
        v_out = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
        onnx_value.append(v_out)

        if mode == 'constant':
            p_node, out = make_node('Pad', inputs=[in_name, 'pads', 'value'], outputs=[out_name], mode='constan')
            onnx_node.append(p_node)
        else:
            p_node, out = make_node('Pad', inputs=[in_name, 'pads'], outputs=[out_name], mode=mode)
            onnx_node.append(p_node)

        return onnx_node, onnx_value, onnx_init