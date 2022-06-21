#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, make_shape_channels_first, get_channels_first_permutation, squeeze_axes
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np

@OpMapper(['GlobalMaxPool1d', 'GlobalMaxPool2d', 'GlobalMaxPool3d'])
class GlobalMaxPool():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        # input , output
        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]

        # added dimensions
        out_shape_temp = [node['out_tensors'][0][0], node['out_tensors'][0][1], 1, 1]
        out_temp_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'temp', NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=out_shape_temp)
        onnx_value.append(out_temp_v)
        # out dimensions
        out_shape = node['out_tensors'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=out_shape)
        onnx_value.append(out_v)

        # attrbute
        data_format = node['attr']['data_format']
        spatial = int(node['node'].layer.__class__.__name__[-2])

        if data_format == 'channels_last':
            # channels last conver weights and input
            x_shape = make_shape_channels_first(x_shape)
            # make channels transpose
            t_x = helper.make_tensor_value_info(node['in_nodes_name'][0] + 't',
                                                NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=x_shape)
            onnx_value.append(t_x)
            tx_node, x = make_node('Transpose', inputs=[x], outputs=[node['in_nodes_name'][0] + 't'],
                                   perm=get_channels_first_permutation(spatial))
            onnx_node.append(tx_node)
        # make Global MaxPool
        gmp_node, x = make_node('GlobalMaxPool',
                                 inputs=[x],
                                 outputs=[node['out_nodes_name'][0] + 'temp']
                                 )
        onnx_node.append(gmp_node)

        # squeeze dimensions
        axes = np.array(squeeze_axes(spatial)).astype(np.int64)
        axes_value = numpy_helper.from_array(axes, name='axes')
        onnx_init.append(axes_value)
        sq_node, _ = make_node('Squeeze', inputs=[x, 'axes'], outputs=node['out_nodes_name'])
        onnx_node.append(sq_node)
        return onnx_node, onnx_value, onnx_init


@OpMapper(['GlobalAvgPool1d', 'GlobalAvgPool2d', 'GlobalAvgPool3d'])
class GlobalAvgPool():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        # input , output
        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]

        # added dimensions
        out_shape_temp = [node['out_tensors'][0][0], node['out_tensors'][0][1], 1, 1]
        out_temp_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'temp', NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=out_shape_temp)
        onnx_value.append(out_temp_v)
        # out dimensions
        out_shape = node['out_tensors'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=out_shape)
        onnx_value.append(out_v)

        # attrbute
        data_format = node['attr']['data_format']
        spatial = int(node['node'].layer.__class__.__name__[-2])

        if data_format == 'channels_last':
            # channels last conver weights and input
            x_shape = make_shape_channels_first(x_shape)
            # make channels transpose
            t_x = helper.make_tensor_value_info(node['in_nodes_name'][0] + 't',
                                                NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=x_shape)
            onnx_value.append(t_x)
            tx_node, x = make_node('Transpose', inputs=[x], outputs=[node['in_nodes_name'][0] + 't'],
                                   perm=get_channels_first_permutation(spatial))
            onnx_node.append(tx_node)
        # make Global MaxPool
        gmp_node, x = make_node('GlobalAveragePool',
                                 inputs=[x],
                                 outputs=[node['out_nodes_name'][0] + 'temp']
                                 )
        onnx_node.append(gmp_node)

        # squeeze dimensions
        axes = np.array(squeeze_axes(spatial)).astype(np.int64)
        axes_value = numpy_helper.from_array(axes, name='axes')
        onnx_init.append(axes_value)
        sq_node, _ = make_node('Squeeze', inputs=[x, 'axes'], outputs=node['out_nodes_name'])
        onnx_node.append(sq_node)
        return onnx_node, onnx_value, onnx_init