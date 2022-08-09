#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import onnx
from onnx import helper, numpy_helper
from collections import OrderedDict
import tensorlayerx as tlx
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node, tlx_act_2_onnx
from tlx2onnx.common import make_shape_channels_first, get_channels_first_permutation,get_channels_last_permutation

@OpMapper(["SubpixelConv2d"])
class SubpixelConv():

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        op_type = "DepthToSpace"
        attr_dict = OrderedDict()
        # get in_node_name out_node_nmae
        x_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_shape = node['out_tensors'][0]

        # get data_type
        data_type = node['dtype']
        tensor_type = NP_TYPE_TO_TENSOR_TYPE[data_type]

        # get cur_node_layer node_index
        layer = node['node'].layer
        layer_name = layer.__class__.__name__
        spatial = int(layer_name[-2])

        # get layer attr
        scale = layer.scale
        data_format = layer.data_format
        attr_dict["blocksize"] = scale

        if data_format == "channels_last":
            permutation = get_channels_first_permutation(spatial)
            x_shape_t = make_shape_channels_first(x_shape)
            transpose_value = helper.make_tensor_value_info(x_name + '_t', tensor_type, shape=x_shape_t)
            onnx_value.append(transpose_value)
            transpose_node, out = make_node('Transpose', inputs=[x_name], outputs=[x_name + '_t'], perm=permutation)
            onnx_node.append(transpose_node)
            depth_to_space, out = make_node(op_type, inputs=[out], outputs=[out + '_t'], **attr_dict)
            onnx_node.append(depth_to_space)
            if node['node'].layer.act is not None:
                act_op = node['node'].layer.act.__class__.__name__
                act_node, out = tlx_act_2_onnx[act_op]([out], [out + '_act'])
                onnx_node.append(act_node)
            permutation = get_channels_last_permutation(spatial)
            transpose_node, out = make_node('Transpose', inputs=[out], outputs=[out_name], perm=permutation)
            onnx_node.append(transpose_node)
            return onnx_node, onnx_value, onnx_init

        elif data_format == 'channels_first':
            if node['node'].layer.act is None:
                depth_to_space, out = make_node(op_type, inputs=[x_name], outputs=[out_name], **attr_dict)
                onnx_node.append(depth_to_space)
                return onnx_node, onnx_value, onnx_init
            else:
                depth_to_space, out = make_node(op_type, inputs=[x_name], outputs=[out_name+ '_act'], **attr_dict)
                onnx_node.append(depth_to_space)
                act_op = node['node'].layer.act.__class__.__name__
                act_node, out = tlx_act_2_onnx[act_op]([out], [out_name])
                onnx_node.append(act_node)
                return onnx_node, onnx_value, onnx_init
        else:
            raise ValueError(
            "Only support 'channels_first' or 'channels_last' data_format mode, but got {}.".format(data_format))

    @classmethod
    def version_11(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        op_type = "DepthToSpace"
        attr_dict = OrderedDict()
        # get in_node_name out_node_nmae
        x_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_shape = node['out_tensors'][0]

        # get data_type
        data_type = node['dtype']
        tensor_type = NP_TYPE_TO_TENSOR_TYPE[data_type]

        # get cur_node_layer node_index
        layer = node['node'].layer
        layer_name = layer.__class__.__name__
        spatial = int(layer_name[-2])

        # get layer attr
        scale = layer.scale
        data_format = layer.data_format
        attr_dict["blocksize"] = scale
        if tlx.BACKEND in ["tensorflow", "mindspore"]:
            attr_dict["mode"] = "DCR"
        elif tlx.BACKEND in ["torch", "paddle"]:
            attr_dict["mode"] = "CRD"

        if data_format == "channels_last":
            permutation = get_channels_first_permutation(spatial)
            x_shape_t = make_shape_channels_first(x_shape)
            transpose_value = helper.make_tensor_value_info(x_name + '_t', tensor_type, shape=x_shape_t)
            onnx_value.append(transpose_value)
            transpose_node, out = make_node('Transpose', inputs=[x_name], outputs=[x_name + '_t'], perm=permutation)
            onnx_node.append(transpose_node)
            depth_to_space, out = make_node(op_type, inputs=[out], outputs=[out + '_t'], **attr_dict)
            onnx_node.append(depth_to_space)
            if node['node'].layer.act is not None:
                act_op = node['node'].layer.act.__class__.__name__
                act_node, out = tlx_act_2_onnx[act_op]([out], [out + '_act'])
                onnx_node.append(act_node)
            permutation = get_channels_last_permutation(spatial)
            transpose_node, out = make_node('Transpose', inputs=[out], outputs=[out_name], perm=permutation)
            onnx_node.append(transpose_node)
            return onnx_node, onnx_value, onnx_init

        elif data_format == 'channels_first':
            if node['node'].layer.act is None:
                depth_to_space, out = make_node(op_type, inputs=[x_name], outputs=[out_name], **attr_dict)
                onnx_node.append(depth_to_space)
                return onnx_node, onnx_value, onnx_init
            else:
                depth_to_space, out = make_node(op_type, inputs=[x_name], outputs=[out_name+ '_act'], **attr_dict)
                onnx_node.append(depth_to_space)
                act_op = node['node'].layer.act.__class__.__name__
                act_node, out = tlx_act_2_onnx[act_op]([out], [out_name])
                onnx_node.append(act_node)
                return onnx_node, onnx_value, onnx_init
        else:
            raise ValueError(
                "Only support 'channels_first' or 'channels_last' data_format mode, but got {}.".format(data_format))