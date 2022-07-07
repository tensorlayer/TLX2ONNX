#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node, get_channels_last_permutation, get_channels_first_permutation
import numpy as np


@OpMapper(["UpSampling2d"])
class UpSampling2d():
    # suppport v1-v13

    @classmethod
    def version_1(cls, node, **kwargs):
        # Get inputs outputs
        mode = {'nearest': 'nearest', 'bilinear': 'linear', 'bicubic': 'cubic'}
        onnx_node, onnx_value, onnx_init = [], [], []
        x_name = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        layer = node['node'].layer
        scale = layer.scale
        method = layer.method
        data_format = layer.data_format
        spatial = int(node['node'].layer.__class__.__name__[-2])

        # Method used
        if method not in ['bilinear', 'nearest', 'bicubic'] or method == 'area':
            raise Exception('Sampling methods nearest, bilinear, and bicubic are supported.')
        # Scale used
        scales = np.array([1.0, 1.0, scale[0], scale[1]], dtype=np.float32)
        scales_value = numpy_helper.from_array(scales, name=layer.name + 'scales')
        onnx_init.append(scales_value)
        # Make resize node
        if data_format == 'channels_first':
            out_v = helper.make_tensor_value_info(out_name, dtype, out_shape)
            onnx_value.append(out_v)
            out_node, _ = make_node('Resize', inputs=[x_name, '', layer.name + 'scales'], outputs=[out_name], mode=mode[method])
            onnx_node.append(out_node)
        else:
            tx_node, out = make_node('Transpose', inputs=[x_name], outputs=[x_name + 't'],
                                   perm=get_channels_first_permutation(spatial))
            onnx_node.append(tx_node)
            rx_node, out = make_node('Resize', inputs=[out, '', layer.name + 'scales'], outputs=[x_name + 's'], mode=mode[method])
            onnx_node.append(rx_node)
            tout_node, _ = make_node('Transpose', inputs=[out], outputs=[out_name], perm=get_channels_last_permutation(spatial))
            onnx_node.append(tout_node)
        return onnx_node, onnx_value, onnx_init


@OpMapper(["DownSampling2d"])
class DownSampling2d():
    # suppport v1-v13

    @classmethod
    def version_1(cls, node, **kwargs):
        # Get inputs outputs
        mode = {'nearest': 'nearest', 'bilinear': 'linear', 'bicubic': 'cubic'}
        onnx_node, onnx_value, onnx_init = [], [], []
        x_name = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_name = node['out_nodes_name'][0]
        out_shape = node['out_tensors'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        layer = node['node'].layer
        scale = layer.scale
        method = layer.method
        data_format = layer.data_format
        spatial = int(node['node'].layer.__class__.__name__[-2])

        # Method used
        if method not in ['bilinear', 'nearest', 'bicubic'] or method == 'area':
            raise Exception('Sampling methods nearest, bilinear, and bicubic are supported.')
        # Scale used
        scale = [1.0 / scale[0], 1.0 / scale[1]]
        scales = np.array([1.0, 1.0, scale[0], scale[1]], dtype=np.float32)
        scales_value = numpy_helper.from_array(scales, name=layer.name + 'scales')
        onnx_init.append(scales_value)
        # Make resize node
        if data_format == 'channels_first':
            out_v = helper.make_tensor_value_info(out_name, dtype, out_shape)
            onnx_value.append(out_v)
            out_node, _ = make_node('Resize', inputs=[x_name, '', layer.name + 'scales'], outputs=[out_name], mode=mode[method])
            onnx_node.append(out_node)
        else:
            tx_node, out = make_node('Transpose', inputs=[x_name], outputs=[x_name + 't'],
                                   perm=get_channels_first_permutation(spatial))
            onnx_node.append(tx_node)
            rx_node, out = make_node('Resize', inputs=[out, '', layer.name + 'scales'], outputs=[x_name + 's'], mode=mode[method])
            onnx_node.append(rx_node)
            tout_node, _ = make_node('Transpose', inputs=[out], outputs=[out_name], perm=get_channels_last_permutation(spatial))
            onnx_node.append(tout_node)
        return onnx_node, onnx_value, onnx_init