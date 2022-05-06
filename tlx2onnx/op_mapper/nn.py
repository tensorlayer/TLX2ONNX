#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from collections import OrderedDict
import tensorlayerx as tlx
from .datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np
from .op_mapper import OpMapper
from ..common import make_node


#TODO : CHANGE ACT CONVERTER TO ACTIVATION.PY
def convert_tlx_relu(inputs, outputs, name):

    return make_node("Relu", inputs = inputs, outputs = outputs, name = name)

tlx_act_2_onnx = {
    "ReLU" :  convert_tlx_relu,
    tlx.ELU : "Elu",
    tlx.Tanh : "Tanh",
    tlx.Sigmoid : "Sigmoid",
    tlx.LeakyReLU : "LeakyRelu",
    tlx.Softplus : "Softplus",
    tlx.ReLU6 : "Relu6",
}

def make_shape_channels_first(shape):
    """Makes a (N, ..., C) shape into (N, C, ...)."""

    return shape[:1] + shape[-1:] + shape[1:-1]


def make_shape_channels_last(shape):
    """Makes a (N, C, ...) shape into (N, ..., C)."""

    return shape[:1] + shape[1:-1] + shape[1:2]

def get_channels_first_permutation(spatial):
    """Returns a permutation to make a (N, ..., C) array into (N, C, ...)."""

    return [0, spatial + 1] + list(range(1, spatial + 1))

def convert_padding(pad_mode, input_shape, output_shape, kernel_shape, strides, dilations, spatial, data_format):

    if pad_mode == "SAME":
        pads = [0] * (spatial * 2)
        input_shape = make_shape_channels_first(input_shape)
        output_shape = make_shape_channels_first(output_shape)
        if data_format == "channels_last":
            input_shape = make_shape_channels_first(input_shape)
            output_shape = make_shape_channels_first(output_shape)

        if any(input_shape[i + 2] == -1 or output_shape[i + 2] == -1 for i in range(spatial)):

            auto_pad = "SAME_UPPER"

            return  auto_pad

        for i in range(spatial):
            pad = (
                (output_shape[i + 2] - 1) * strides[i]
                + dilations[i] * (kernel_shape[i] - 1) + 1
                - input_shape[i + 2]
            )
            pad = max(pad, 0)
            pads[i] = pad // 2
            pads[i + spatial] = pad - pad // 2

        return pads

    elif pad_mode == "VALID":
        auto_pad = "VALID"
        return auto_pad

def convert_input(x,  spatial, data_format, in_node_index):
    x_name = in_node_index
    x_numpy = tlx.convert_to_numpy(x)
    x_dtype = x_numpy.dtype
    x_shape = x_numpy.shape
    x_tensor_type = NP_TYPE_TO_TENSOR_TYPE[x_dtype]
    if data_format == 'channels_last':
        permutation = get_channels_first_permutation(spatial)
        x_shape = np.transpose(x_shape, permutation)

    return helper.make_tensor_value_info(x_name, x_tensor_type, x_shape)

def convert_w(w, raw_kernel_shape, spatial, w_name):
    w_numpy = tlx.convert_to_numpy(w)
    w_shape = w_numpy.shape
    if w_shape == raw_kernel_shape:
        w_shape = w_shape[-1:] + w_shape[-2:-1] + w_shape[0:spatial]
    return numpy_helper.from_array(w_numpy.reshape(w_shape), name=w_name)

def convert_b(b, b_name):
    b_numpy = tlx.convert_to_numpy(b)
    return numpy_helper.from_array(b_numpy, name=b_name)


@OpMapper(["Conv1d", "Conv2d", "Conv3d"])
class Conv():

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

        #### get in_nodes node_index
        in_node = node.in_nodes[0]
        in_node_index = str(in_node.node_index)
        #### get cur_node_layer node_index
        cur_node_index = str(node.node_index)
        layer = node.layer
        layer_type = layer.__class__.__name__

        #### get in_tensors
        in_tensors = node.in_tensors[0]

        #### get out_tensors
        out_tensors = node.out_tensors[0]

        #### get layer_param
        layer_param = layer.all_weights

        #### get conv spatial
        spatial = int(layer_type[-2])

        #### get layer_act
        layer_act = layer.act.__class__.__name__
        # layer_act = "ReLU"

        #### get conv raw filter_shape
        raw_kernel_shape = layer.filter_shape

        #### conv inputs
        if len(layer_param) == 1:
            w = layer_param[0]
            b = None
        elif len(layer_param) == 2:
            w = layer_param[0]
            b = layer_param[1]

        #### insert conv attr
        attr_dict["kernel_shape"] = layer.kernel_size
        attr_dict["dilations"] = layer.dilation
        attr_dict["strides"] = layer.stride
        data_format = layer.data_format
        pads = convert_padding(
            layer.padding, in_tensors.shape, out_tensors.shape, attr_dict["kernel_shape"], attr_dict["strides"],
            attr_dict["dilations"], spatial, data_format
        )
        if isinstance(pads, str):
            attr_dict["auto_pad"] = pads
        else:
            attr_dict["pads"] = pads
        attr_dict["group"] = 1
        attr_dict["outputs"] = [cur_node_index]
        attr_dict["name"] = node.node_name

        #### convert x
        x_value_info = convert_input(in_tensors, spatial, data_format, in_node_index)
        onnx_value.append(x_value_info)

        #### convert w
        w_name = cur_node_index + '_w'
        w_onnx_init = convert_w(w, raw_kernel_shape, spatial, w_name)
        onnx_init.append(w_onnx_init)
        attr_dict["inputs"] = [in_node_index, w_name]

        #### convert b
        if b is not None:
            b_name = cur_node_index + '_b'
            b_onnx_init = convert_b(b, b_name)
            onnx_init.append(b_onnx_init)
            attr_dict["inputs"] = [in_node_index, w_name, b_name]

        #### make act node
        if layer_act is not None:
            act_convert = tlx_act_2_onnx[layer_act]
            act_input = cur_node_index + "_act"
            act_out = cur_node_index
            #### 如果layer存在act，需要新增一个act node 和 对应act输入的act input info， 并且要更新 conv的outputs 为 act的inputs， 此时act的outputs是整个layer的outputs
            attr_dict["outputs"] = [act_input]
            act_node = act_convert([act_input], [act_out], node.node_name + "_act")
            out_tensors_numpy = tlx.convert_to_numpy(out_tensors)
            out_tensors_dtype = out_tensors_numpy.dtype
            out_tensors_shape = out_tensors_numpy.shape
            out_tensors_type = NP_TYPE_TO_TENSOR_TYPE[out_tensors_dtype]
            if data_format == 'channels_last':
                permutation = get_channels_first_permutation(spatial)
                out_tensors_shape = np.transpose(out_tensors_shape, permutation)
            act_input_value_info = helper.make_tensor_value_info(act_input, out_tensors_type, out_tensors_shape)
            onnx_value.append(act_input_value_info)
            onnx_node.append(act_node)

        #### make conv node
        conv_node = helper.make_node(Op_name, **attr_dict)
        onnx_node.append(conv_node)

        return onnx_node, onnx_value, onnx_init

    @classmethod
    def version_1(cls, node, **kwargs):

        return cls.any_version(node, 1, **kwargs)


    @classmethod
    def version_11(cls, node, **kwargs):
        # No change.
        return cls.any_version( node, 11, **kwargs)

    @classmethod
    def version_13(cls, node, **kwargs):
        # Signature change for operator Unsqueeze.
        return cls.any_version(node, 13, **kwargs)


