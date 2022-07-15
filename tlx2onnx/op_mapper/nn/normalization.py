#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from tlx2onnx.op_mapper.op_mapper import OpMapper
from tlx2onnx.common import make_node, to_numpy, make_shape_channels_first, make_shape_channels_last, \
    get_channels_first_permutation, get_channels_last_permutation


@OpMapper(['BatchNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d'])
class BatchNorm():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        # input , output, data_format
        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]

        out_shape = node['out_tensors'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=node['out_tensors'][0])
        onnx_value.append(out_v)

        data_format = node['attr']['data_format']
        spatial = int(node['node'].layer.__class__.__name__[-2])
        # get parameters
        beta_name = node['node'].layer.name + '/beta'
        beta_weight = numpy_helper.from_array(arr=to_numpy(node['node'].layer.gamma), name=beta_name)
        onnx_init.append(beta_weight)

        gamma_name = node['node'].layer.name + '/gamma'
        gamma_weight = numpy_helper.from_array(arr=to_numpy(node['node'].layer.beta), name=gamma_name)
        onnx_init.append(gamma_weight)

        mean_name = node['node'].layer.name + '/mean'
        mean_weight = numpy_helper.from_array(arr=to_numpy(node['node'].layer.moving_mean), name=mean_name)
        onnx_init.append(mean_weight)

        var_name = node['node'].layer.name + '/var'
        var_weight = numpy_helper.from_array(arr=to_numpy(node['node'].layer.moving_var), name=var_name)
        onnx_init.append(var_weight)

        if data_format == 'channels_last':
            # channels last conver weights and input
            x_shape = make_shape_channels_first(x_shape)
            out_temp_shape = make_shape_channels_first(out_shape)
            # make channels transpose
            t_x = helper.make_tensor_value_info(node['in_nodes_name'][0] + 't',
                                                NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=x_shape)
            onnx_value.append(t_x)
            tx_node, x = make_node('Transpose', inputs=[x], outputs=[node['in_nodes_name'][0] + 't'],
                                   perm=get_channels_first_permutation(spatial))
            onnx_node.append(tx_node)
            # make batch normalization
            out_temp = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'bn',
                                                     NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=out_temp_shape)
            onnx_value.append(out_temp)
            bn_node, out = make_node('BatchNormalization',
                                     inputs=[node['in_nodes_name'][0] + 't', beta_name, gamma_name, mean_name, var_name],
                                     outputs=[node['out_nodes_name'][0] + 'bn']
                                     )
            onnx_node.append(bn_node)
            # make channels transpose
            t_out = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=out_shape)
            onnx_value.append(t_out)
            tout_node, _ = make_node('Transpose', inputs=[out], outputs=node['out_nodes_name'], perm=get_channels_last_permutation(spatial))
            onnx_node.append(tout_node)


        elif data_format == 'channels_first':
            bn_node, out = make_node('BatchNormalization',
                                     inputs=[x, beta_name, gamma_name, mean_name, var_name],
                                     outputs=node['out_nodes_name']
                                     )
            onnx_node.append(bn_node)
        return onnx_node, onnx_value, onnx_init


@OpMapper(['LayerNorm'])
class LayerNorm():
    # supports v17

    @classmethod
    def version_17(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        # input , output, data_format
        x = node['in_nodes_name'][0]
        x_shape = node['in_tensors'][0]

        out_shape = node['out_tensors'][0]
        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                              shape=node['out_tensors'][0])
        onnx_value.append(out_v)

        spatial = 2
        # get parameters
        beta_name = node['node'].layer.name + '/beta'
        beta_weight = numpy_helper.from_array(arr=to_numpy(node['node'].layer.beta), name=beta_name)
        onnx_init.append(beta_weight)

        gamma_name = node['node'].layer.name + '/gamma'
        gamma_weight = numpy_helper.from_array(arr=to_numpy(node['node'].layer.gamma), name=gamma_name)
        onnx_init.append(gamma_weight)

        epsilon = node['node'].layer.epsilon

        # if data_format == 'channels_last':
            # channels last conver weights and input
        x_shape = make_shape_channels_first(x_shape)
        out_temp_shape = make_shape_channels_first(out_shape)
        # make channels transpose
        t_x = helper.make_tensor_value_info(node['in_nodes_name'][0] + 't',
                                            NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=x_shape)
        onnx_value.append(t_x)
        tx_node, x = make_node('Transpose', inputs=[x], outputs=[node['in_nodes_name'][0] + 't'],
                               perm=get_channels_first_permutation(spatial))
        onnx_node.append(tx_node)
        # make batch normalization
        out_temp = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'bn',
                                                 NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=out_temp_shape)
        onnx_value.append(out_temp)
        ln_node, out = make_node('LayerNormalization',
                                 inputs=[node['in_nodes_name'][0] + 't', beta_name, gamma_name],
                                 outputs=[node['out_nodes_name'][0] + 'bn'], epsilon=epsilon
                                 )
        onnx_node.append(ln_node)
        # make channels transpose
        t_out = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']], shape=out_shape)
        onnx_value.append(t_out)
        tout_node, _ = make_node('Transpose', inputs=[out], outputs=node['out_nodes_name'], perm=get_channels_last_permutation(spatial))
        onnx_node.append(tout_node)
        return onnx_node, onnx_value, onnx_init