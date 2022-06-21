#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from ...common import tlx_act_2_onnx, convert_padding, make_shape_channels_first, convert_w, \
    get_channels_last_permutation, get_channels_first_permutation

@OpMapper(['ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'])
class ConvTranspose():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]
        out_shape = node['out_tensors'][0]
        spatial = int(node['node'].layer.__class__.__name__[-2])

        y = node['node'].layer.name + '/weights'
        weights_value = node['node'].layer.filters

        attr_dict = {}
        attr_dict['dilations'] = dilations = node['attr']['dilation']
        attr_dict['kernel_shape'] = kernel_shape = node['attr']['kernel_size']
        attr_dict['strides'] = strides = node['attr']['stride']
        pads = node['attr']['padding']
        data_format = node['attr']['data_format']

        if data_format == 'channels_last':
            # channels last conver weights and input
            x_shape = make_shape_channels_first(x_shape)
            out_temp_shape = make_shape_channels_first(out_shape)
            weights_value = convert_w(weights_value, data_format, spatial)
            t_x = helper.make_tensor_value_info(node['in_nodes_name'][0] + 't', NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=x_shape)
            onnx_value.append(t_x)
            tx_node, x = make_node('Transpose', inputs=[x], outputs=[node['in_nodes_name'][0] + 't'], perm=get_channels_first_permutation(spatial))
            onnx_node.append(tx_node)


        # Build weights
        weights = numpy_helper.from_array(arr=to_numpy(weights_value), name=y)
        onnx_init.append(weights)
        # Build padding
        pads = convert_padding(
            pads, x_shape, out_shape, kernel_shape, strides,
            dilations, spatial, data_format
        )
        if isinstance(pads, str):
            attr_dict["auto_pad"] = pads
        else:
            attr_dict["pads"] = pads

        if node['node'].layer.b_init is not None:
            b = numpy_helper.from_array(arr=to_numpy(node['node'].layer.biases), name=node['node'].layer.name + '/b')
            onnx_init.append(b)
            b_name = node['node'].layer.name + '/b'
            input_list = [x, y, b_name]
        else:
            input_list = [x, y]

        if data_format == 'channels_first':
            if node['node'].layer.act is not None:
                # Build ConvTranspose
                de_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'de', NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                                     shape=out_shape)
                onnx_value.append(de_v)
                ct_node, out = make_node('ConvTranspose', inputs=input_list,
                                        outputs=[node['out_nodes_name'][0] + 'de'], **attr_dict)
                onnx_node.append(ct_node)

                act_op = node['node'].layer.act.__class__.__name__
                out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                                      shape=out_shape)
                onnx_value.append(out_v)
                # Using Opmapper
                act_node, _ = tlx_act_2_onnx[act_op]([out], node['out_nodes_name'], node['node'].layer.act)
                onnx_node.append(act_node)
            else:
                out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                                      shape=out_shape) #
                onnx_value.append(out_v)
                ct_node, out = make_node('ConvTranspose', inputs=input_list,
                                        outputs=node['out_nodes_name'], **attr_dict)
                onnx_node.append(ct_node)
        elif data_format == 'channels_last':
            if node['node'].layer.act is not None:
                # Build ConvTranspose
                ct_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'ct', NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                                     shape=out_temp_shape)
                onnx_value.append(ct_v)
                ct_node, out = make_node('ConvTranspose', inputs=input_list,
                                        outputs=[node['out_nodes_name'][0] + 'ct'], **attr_dict)
                onnx_node.append(ct_node)

                act_op = node['node'].layer.act.__class__.__name__
                act_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'a', NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                                      shape=out_temp_shape)
                onnx_value.append(act_v)
                # Using Opmapper
                act_node, out = tlx_act_2_onnx[act_op]([out], [node['out_nodes_name'][0] + 'a'], node['node'].layer.act)
                onnx_node.append(act_node)
            else:
                out_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'ct', NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                                      shape=out_temp_shape)
                onnx_value.append(out_v)
                o_node, out = make_node('ConvTranspose', inputs=input_list,
                                        outputs=[node['out_nodes_name'][0] + 'ct'], **attr_dict)
                onnx_node.append(o_node)

            t_out = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=out_shape)
            onnx_value.append(t_out)
            tout_node, _ = make_node('Transpose', inputs=[out], outputs=node['out_nodes_name'], perm=get_channels_last_permutation(spatial))
            onnx_node.append(tout_node)
        else:
            raise ValueError("Only support 'channels_first' or 'channels_last' data_format mode, but got {}.".format(data_format))

        return onnx_node, onnx_value, onnx_init


