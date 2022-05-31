#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from collections import OrderedDict
import tensorlayerx as tlx
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node


def convert_tlx_relu(inputs, outputs, name = None):
    opsets = OpMapper.OPSETS['ReLU']
    map_func, kw= opsets[1]
    kw = {"inputs" : inputs,
          "outputs" : outputs}
    return map_func(node = None, **kw)

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

def get_channels_last_permutation(spatial):
    """Returns a permutation to make a (N, C, ...) array into (N, ..., C)."""

    return [0] + list(range(2, spatial+2)) + [1]

def convert_padding(padding, input_shape, output_shape, kernel_shape, strides, dilations, spatial, data_format):
    if isinstance(padding, str):
        if padding == "SAME":
            pads = [0] * (spatial * 2)
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

        elif padding == "VALID":
            auto_pad = "VALID"
            return auto_pad
    elif isinstance(padding, int):
        pads = [padding] * spatial * 2
        return pads
    elif isinstance(padding, tuple):
        return list(padding) * 2

def convert_w(w, data_format, spatial, w_name):
    w = tlx.convert_to_numpy(w)
    w_shape = w.shape
    if tlx.BACKEND == 'tensorflow':
        w_shape = w_shape[-1:] + w_shape[-2:-1] + w_shape[0:spatial]
        return numpy_helper.from_array(w.reshape(w_shape), name=w_name)
    elif tlx.BACKEND == 'mindspore':
        if spatial == 2 and data_format == 'channels_last':
            w_shape = w_shape[0] + w[-1:] + w[1:3]
            return numpy_helper.from_array(w.reshape(w_shape), name=w_name)
    return numpy_helper.from_array(w, name=w_name)

def convert_b(b, b_name):
    b = tlx.convert_to_numpy(b)
    return numpy_helper.from_array(b, name=b_name)


@OpMapper(["Conv1d", "Conv2d", "Conv3d"])
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
        attr_dict["kernel_shape"] = node['attr']['kernel_size']
        attr_dict["dilations"] = node['attr']['dilation']
        attr_dict["strides"] = node['attr']['stride']
        data_format = node['attr']['data_format']
        paddding = node['attr']['padding']
        attr_dict["group"] = 1
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


