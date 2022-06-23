#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper
from collections import OrderedDict
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node
from tlx2onnx.common import make_shape_channels_first, get_channels_first_permutation,tlx_act_2_onnx,get_channels_last_permutation
from tlx2onnx.common import convert_padding


@OpMapper(["MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d"])
class MaxPool():
    # suppport v1-v11

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        attr_dict = OrderedDict()
        # get in_node_name out_node_nmae
        x_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_shape = node['out_tensors'][0]

        #### get data_type
        data_type = node['dtype']
        tensor_type = NP_TYPE_TO_TENSOR_TYPE[data_type]

        # get cur_node_layer node_index
        layer = node['node'].layer
        layer_name = layer.__class__.__name__
        spatial = int(layer_name[-2])
        layer_type = layer_name[:7]
        if layer_type == "MaxPool":
            Op_name = "MaxPool"
        elif layer_type == "AvgPool":
            Op_name = "AveragePool"

        # insert pool attr
        kernel_size = node['attr']['kernel_size']
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        attr_dict["kernel_shape"] = kernel_size
        strides = node['attr']['stride']
        if isinstance(strides, int):
            strides = [strides]
        attr_dict["strides"] = strides
        data_format = node['attr']['data_format']
        paddding = node['attr']['padding']

        # convert padding
        pads = convert_padding(
            paddding, x_shape, out_shape, attr_dict["kernel_shape"], attr_dict["strides"],
            None, spatial, data_format
        )
        if isinstance(pads, str):
            attr_dict["auto_pad"] = pads
        else:
            attr_dict["pads"] = pads

        if data_format == 'channels_last':
            permutation = get_channels_first_permutation(spatial)
            x_shape_t = make_shape_channels_first(x_shape)
            # insert transpose op: NHWC -> NCHW
            transpose_value = helper.make_tensor_value_info(x_name+'_t', tensor_type, shape=x_shape_t)
            onnx_value.append(transpose_value)
            transpose_node, out = make_node('Transpose', inputs=[x_name], outputs=[x_name+'_t'], perm = permutation)
            onnx_node.append(transpose_node)

            attr_dict["inputs"] = [out]
            attr_dict["outputs"] = [out+'_t']
            maxpool_node, out = make_node(Op_name, **attr_dict)
            onnx_node.append(maxpool_node)
            out_shape_t = make_shape_channels_first(out_shape)
            maxpool_value = helper.make_tensor_value_info(out, tensor_type, shape=out_shape_t)
            onnx_value.append(maxpool_value)

            # insert transpose op: NCHW -> NHWC
            permutation = get_channels_last_permutation(spatial)
            transpose_node, out = make_node('Transpose', inputs=[out], outputs=[out_name], perm=permutation)
            onnx_node.append(transpose_node)
            transpose_value = helper.make_tensor_value_info(out_name, tensor_type, shape=out_shape)
            onnx_value.append(transpose_value)
            return onnx_node, onnx_value, onnx_init

        elif data_format == 'channels_first':

            attr_dict["inputs"] = [x_name]
            attr_dict["outputs"] = [out_name]
            maxpool_node, out = make_node(Op_name, **attr_dict)
            onnx_node.append(maxpool_node)
            maxpool_value = helper.make_tensor_value_info(out, tensor_type, out_shape)
            onnx_value.append(maxpool_value)
            return onnx_node, onnx_value, onnx_init

        else:
            raise ValueError("Only support 'channels_first' or 'channels_last' data_format mode, but got {}.".format(data_format))
