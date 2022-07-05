#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from collections import OrderedDict
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node, convert_w, convert_padding, to_numpy
from tlx2onnx.common import make_shape_channels_first, get_channels_first_permutation,tlx_act_2_onnx,get_channels_last_permutation

@OpMapper(["DepthwiseConv2d"])
class DepthwiseConv():
    # suppport v1-v13

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node, onnx_value, onnx_init = [], [], []
        depth_dict = OrderedDict()
        point_dict = OrderedDict()

        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        # get input output
        x_name = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_shape = node['out_tensors'][0]
        out_name = node['out_nodes_name'][0]

        # get common operation
        layer = node['node'].layer
        spatial = int(layer.__class__.__name__[-2])
        act_op = layer.act.__class__.__name__
        data_format = layer.data_format

        # trainable weights
        depth_filters = layer.filters
        depth_name = layer.name + '_depth_w'
        point_filter = layer.point_filter
        point_name = layer.name + '_point_w'
        depth_init = convert_w(depth_filters, data_format, spatial, depth_name)
        onnx_init.append(depth_init)
        point_init = convert_w(point_filter, data_format, spatial, point_name)
        onnx_init.append(point_init)
        if layer.b_init:
            b_name = layer.name + '_b'
            b = numpy_helper.from_array(arr=to_numpy(node['node'].layer.b), name=b_name)
            onnx_init.append(b)

        # constant value
        depth_dict['kernel_shape'] = kernel_size = layer.kernel_size
        point_dict['kernel_shape'] = [1, 1]
        depth_dict['strides'] = stride = layer.stride
        padding = layer.padding
        depth_dict['dilations'] = dilation = layer.dilation
        depth_multiplier = layer.depth_multiplier
        in_channels = layer.in_channels


        ####convert padding
        if data_format == 'channels_last':
            depth_shape = out_shape[0:3] + [x_shape[3]]
        else:
            depth_shape = x_shape[0:2] + out_shape[2:]
        depth_pads = convert_padding(padding, x_shape, depth_shape, kernel_size, stride, dilation, spatial, data_format)
        point_pads = convert_padding(padding, depth_shape, out_shape, (1, 1), (1, 1), (1, 1), spatial, data_format)
        if isinstance(depth_pads, str):
            depth_dict['auto_pad'] = depth_pads
        else:
            depth_dict['pads'] = depth_pads

        if isinstance(point_pads, str):
            point_dict['auto_pad'] = point_pads
        else:
            point_dict['pads'] = point_pads

        if data_format == 'channels_last':
            permutation = get_channels_first_permutation(spatial)
            x_t = make_shape_channels_first(x_shape)

            t_value = helper.make_tensor_value_info(x_name+'_t', dtype, shape=x_t)
            onnx_value.append(t_value)
            t_node, out = make_node('Transpose', inputs=[x_name], outputs=[x_name+'_t'], perm = permutation)
            onnx_node.append(t_node)

            # make depthwise
            depth_shape_temp = out_shape[0:3] + [x_t[1]]
            depthwise_out_shape = make_shape_channels_first(depth_shape_temp)
            depth_out = helper.make_tensor_value_info(layer.name + 'depth_out', dtype, shape=depthwise_out_shape)
            onnx_value.append(depth_out)
            depth_node, out = make_node('Conv', inputs=[out, depth_name], outputs=[layer.name + 'depth_out'],
                                        group= in_channels, **depth_dict)
            onnx_node.append(depth_node)

            # make pointwise
            point_out = helper.make_tensor_value_info(layer.name + 'point_out', dtype, shape=make_shape_channels_first(out_shape))
            onnx_value.append(point_out)
            # make bias
            if layer.b_init:
                point_inputs = [out, point_name, b_name]
            else:
                point_inputs = [out, point_name]
            point_node, out = make_node('Conv', inputs=point_inputs, outputs=[layer.name + 'point_out'],
                                        group=1, **point_dict)
            onnx_node.append(point_node)

            # make activation
            if node['node'].layer.act is not None:
                act_out = helper.make_tensor_value_info(layer.name + 'act_out', dtype, shape=make_shape_channels_first(out_shape))
                onnx_value.append(act_out)
                act_node, out = tlx_act_2_onnx[act_op]([out], [layer.name + 'act_out'], layer.act)
                onnx_node.append(act_node)

            # Convert the result to channel last
            permutation = get_channels_last_permutation(spatial)
            transpose_node, out = make_node('Transpose', inputs=[out], outputs=[out_name], perm=permutation)
            onnx_node.append(transpose_node)
            transpose_value = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
            onnx_value.append(transpose_value)

        elif data_format == 'channels_first':
            # make depthwise
            depthwise_out_shape = x_shape[0:2] + out_shape[2:]
            depth_out = helper.make_tensor_value_info(layer.name + 'depth_out', dtype, shape=depthwise_out_shape)
            onnx_value.append(depth_out)
            depth_node, out = make_node('Conv', inputs=[x_name, depth_name], outputs=[layer.name + 'depth_out'],
                                        group=in_channels, **depth_dict)
            onnx_node.append(depth_node)

            # make activation
            if node['node'].layer.act is not None:
                # make pointwise
                point_out = helper.make_tensor_value_info(layer.name + 'point_out', dtype,
                                                          shape=out_shape)
                onnx_value.append(point_out)
                # make bias
                if layer.b_init:
                    point_inputs = [out, point_name, b_name]
                else:
                    point_inputs = [out, point_name]
                point_node, out = make_node('Conv', inputs=point_inputs, outputs=[layer.name + 'point_out'],
                                            group=1, **point_dict)
                onnx_node.append(point_node)
                act_out = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
                onnx_value.append(act_out)
                act_node, out = tlx_act_2_onnx[act_op]([out], [out_name], layer.act)
                onnx_node.append(act_node)
            else:
                # make pointwise
                point_out = helper.make_tensor_value_info(out_name, dtype, shape=out_shape)
                onnx_value.append(point_out)
                # make bias
                if layer.b_init:
                    point_inputs = [out, point_name, b_name]
                else:
                    point_inputs = [out, point_name]
                point_node, out = make_node('Conv', inputs=point_inputs, outputs=[out_name],
                                            group=1, **point_dict)
                onnx_node.append(point_node)
        else:
            raise ValueError("Only support 'channels_first' or 'channels_last' data_format mode, but got {}.".format(data_format))
        return onnx_node, onnx_value, onnx_init
