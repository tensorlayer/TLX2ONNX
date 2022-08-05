#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from onnx import helper, numpy_helper
from collections import OrderedDict
import tensorlayerx as tlx
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node
from tlx2onnx.common import make_shape_channels_first, get_channels_first_permutation,get_channels_last_permutation
from tlx2onnx.common import convert_padding, convert_w, tlx_act_2_onnx, convert_b

@OpMapper(["GroupConv2d"])
class Conv():
    # suppport v1-v13

    @classmethod
    def any_version(cls, node, opset, **kwargs):
        """
        Parameters
        ----------
        node:node dict {node: node,
                            in_tensors: node inputs,
                            out_tensors: node outputs,
                            in_nodes_name: node inputs name,
                            out_nodes_name: node outputs name}
        Returns
        -------
        """
        Op_name = 'Conv'
        onnx_node, onnx_value, onnx_init = [], [], []
        attr_dict = OrderedDict()

        #### get data_type
        data_type = node['dtype']
        tensor_type = NP_TYPE_TO_TENSOR_TYPE[data_type]
        #### get in_node_name out_node_nmae
        x_name = node['in_nodes_name'][0]
        out_name = node['out_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_shape = node['out_tensors'][0]

        #### get cur_node_layer node_index
        layer = node['node'].layer
        layer_type = layer.__class__.__name__
        spatial = int(layer_type[-2])
        node_name = layer.name
        #### get layer_param
        layer_param = layer.all_weights

        #### get layer_act_type
        layer_act = layer.act.__class__.__name__

        #### conv inputs
        w = None
        b = None
        if len(layer_param) == 1:
            w = layer_param[0]
        elif len(layer_param) == 2:
            w = layer_param[0]
            b = layer_param[1]

        #### insert conv attr
        kernel_size = node['attr']['kernel_size']
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        attr_dict["kernel_shape"] = kernel_size
        dilations = node['attr']['dilation']
        if isinstance(dilations, int):
            dilations = [dilations,]
        attr_dict["dilations"] = dilations
        strides = node['attr']['stride']
        if isinstance(strides, int):
            strides = [strides]
        attr_dict["strides"] = strides
        data_format = node['attr']['data_format']
        paddding = node['attr']['padding']
        attr_dict["group"] = layer.n_group
        attr_dict["outputs"] = [out_name]

        ####convert padding
        pads = convert_padding(
            paddding, x_shape, out_shape, attr_dict["kernel_shape"], attr_dict["strides"],
            attr_dict["dilations"], spatial, data_format
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
            # convert w
            w_name = node_name + '_w'
            w_init = convert_w(w, data_format, spatial, w_name)
            onnx_init.append(w_init)
            attr_dict["inputs"] = [out, w_name]

            #### convert b
            if b is not None:
                b_name = node_name + '_b'
                b_init = convert_b(b, b_name)
                onnx_init.append(b_init)
                attr_dict["inputs"] = [out, w_name, b_name]

            attr_dict["outputs"] = [out + "_t"]
            conv_node, out = make_node(Op_name, **attr_dict)
            onnx_node.append(conv_node)
            out_shape_t = make_shape_channels_first(out_shape)
            conv_value = helper.make_tensor_value_info(out, tensor_type, shape=out_shape_t)
            onnx_value.append(conv_value)
            # insert transpose op: NCHW -> NHWC  and  insert act node

            if layer_act != 'NoneType':
                act_convert = tlx_act_2_onnx[layer_act]
                act_input = out_name + "_act"
                act_out = out_name
                # insert transpose op
                permutation = get_channels_last_permutation(spatial)
                transpose_node, out = make_node('Transpose', inputs=[out], outputs=[act_input], perm = permutation)
                onnx_node.append(transpose_node)
                transpose_value = helper.make_tensor_value_info(act_input, tensor_type, shape = out_shape)
                onnx_value.append(transpose_value)
                # 如果layer存在act，需要新增一个act node 和 对应act输入的act input info， 并且要更新 conv的outputs 为 act的inputs， 此时act的outputs是整个layer的outputs
                act_node, _ = act_convert([out], [act_out])
                act_input_value_info = helper.make_tensor_value_info(act_out, tensor_type, out_shape)
                onnx_value.append(act_input_value_info)
                onnx_node.append(act_node)
                return onnx_node, onnx_value, onnx_init
            else:
                permutation = get_channels_last_permutation(spatial)
                transpose_node, out = make_node('Transpose', inputs=[out], outputs=[out_name], perm=permutation)
                onnx_node.append(transpose_node)
                transpose_value = helper.make_tensor_value_info(out_name, tensor_type, shape=out_shape)
                onnx_value.append(transpose_value)
                return onnx_node, onnx_value, onnx_init


        elif data_format == 'channels_first':

            #### convert w
            w_name = node_name + '_w'
            w_init = convert_w(w, data_format, spatial, w_name)
            onnx_init.append(w_init)
            attr_dict["inputs"] = [x_name, w_name]

            #### convert b
            if b is not None:
                b_name = node_name + '_b'
                b_init = convert_b(b, b_name)
                onnx_init.append(b_init)
                attr_dict["inputs"] = [x_name, w_name, b_name]

            #### make act node
            if layer_act != 'NoneType':
                act_convert = tlx_act_2_onnx[layer_act]
                act_input = out_name + "_act"
                act_out = out_name
                attr_dict["outputs"] = [act_input]
                conv_node, out = make_node(Op_name, **attr_dict)
                onnx_node.append(conv_node)
                conv_value = helper.make_tensor_value_info(out, tensor_type, shape = out_shape)
                onnx_value.append(conv_value)
                #insert act node
                act_node, out = act_convert([act_input], [act_out])
                act_input_value_info = helper.make_tensor_value_info(out, tensor_type, out_shape)
                onnx_value.append(act_input_value_info)
                onnx_node.append(act_node)
                return onnx_node, onnx_value, onnx_init
            else:
                conv_node, out = make_node(Op_name, **attr_dict)
                onnx_node.append(conv_node)
                conv_value = helper.make_tensor_value_info(out, tensor_type, out_shape)
                onnx_value.append(conv_value)
                return onnx_node, onnx_value, onnx_init
        else:
            raise ValueError("Only support 'channels_first' or 'channels_last' data_format mode, but got {}.".format(data_format))

    @classmethod
    def version_1(cls, node, **kwargs):

        return cls.any_version(node, 1, **kwargs)


    @classmethod
    def version_11(cls, node, **kwargs):

        return cls.any_version( node, 11, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):

        return cls.any_version(node, 13, **kwargs)