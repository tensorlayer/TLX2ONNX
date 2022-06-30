#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, numpy_helper, TensorProto
from .op_mapper import OpMapper
from ..common import make_node, to_numpy
from tlx2onnx.op_mapper.datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np

__all__ = ['Relu', 'LeakyReLU', 'ELU', 'Tanh', 'Sigmoid', 'Softmax', 'Softplus', 'ReLU6', 'PRelu',
           'Mish', 'Swish', 'LeakyReLU6']

@OpMapper(["ReLU"])
class Relu():

    @classmethod
    def version_1(cls, node, **kwargs):
        Op_name = 'Relu'
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None :
            # get in_node_name out_node_name out_tensor_shape
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            r_node, out = make_node(Op_name, inputs=[x_name], outputs=[out_name])
            value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_node.append(r_node)
            onnx_value.append(value)
            return onnx_node, onnx_value, onnx_init
        return make_node(Op_name, **kwargs)


@OpMapper(["LeakyReLU"])
class LeakyReLU():

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None:
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            alpha = node['node'].layer.negative_slope
            l_node, out = make_node('LeakyRelu', inputs=[x_name], outputs=[out_name], alpha=alpha)
            value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_node.append(l_node)
            onnx_value.append(value)
            return onnx_node, onnx_value, onnx_init
        return make_node('LeakyRelu', **kwargs)


@OpMapper(["LeakyReLU6"])
class LeakyReLU6():

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None:
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            alpha = node['node'].layer.alpha
            l_value = helper.make_tensor_value_info(out_name + 'lrelu', NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_value.append(l_value)
            l_node, out = make_node('LeakyRelu', inputs=[x_name], outputs=[out_name + 'lrelu'], alpha=alpha)
            onnx_node.append(l_node)

            value = helper.make_tensor_value_info(out_name, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_value.append(value)
            max = np.array(6).astype(node['dtype'])
            max_value = numpy_helper.from_array(max, name='max')
            onnx_init.append(max_value)
            min_node, out = make_node('Clip', inputs=[out, "", 'max'], outputs=[out_name])
            onnx_node.append(min_node)
            return onnx_node, onnx_value, onnx_init
        return make_node('LeakyRelu', **kwargs)


@OpMapper(["ELU"])
class ELU():

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None:
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            alpha = node['node'].layer.alpha
            e_node, out = make_node('Elu', inputs=[x_name], outputs=[out_name], alpha=alpha)
            value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_node.append(e_node)
            onnx_value.append(value)
            return onnx_node, onnx_value, onnx_init
        return make_node('Elu', **kwargs)


@OpMapper(["Tanh"])
class Tanh():

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None:
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            t_node, out = make_node('Tanh', inputs=[x_name], outputs=[out_name])
            value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_node.append(t_node)
            onnx_value.append(value)
            return onnx_node, onnx_value, onnx_init
        return make_node('Tanh', **kwargs)


@OpMapper(["Sigmoid"])
class Sigmoid():

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None:
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            s_node, out = make_node('Sigmoid', inputs=[x_name], outputs=[out_name])
            value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_node.append(s_node)
            onnx_value.append(value)
            return onnx_node, onnx_value, onnx_init
        return make_node('Sigmoid', **kwargs)


@OpMapper(["Softmax"])
class Softmax():

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None:
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            axis = node['node'].layer.axis
            s_node, out = make_node('Softmax', inputs=[x_name], outputs=[out_name], axis=axis)
            value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_node.append(s_node)
            onnx_value.append(value)
            return onnx_node, onnx_value, onnx_init
        return make_node('Softmax', **kwargs)


@OpMapper(["Softplus"])
class Softplus():

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []
        if node is not None:
            x_name = node['in_nodes_name'][0]
            out_name = node['out_nodes_name'][0]
            out_shape = node['out_tensors'][0]
            s_node, out = make_node('Softplus', inputs=[x_name], outputs=[out_name])
            value = helper.make_tensor_value_info(out, NP_TYPE_TO_TENSOR_TYPE[node['dtype']], out_shape)
            onnx_node.append(s_node)
            onnx_value.append(value)
            return onnx_node, onnx_value, onnx_init
        return make_node('Softplus', **kwargs)


@OpMapper(["ReLU6"])
class ReLU6():

    @classmethod
    def version_1(cls, node, **kwargs):

        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]

        relu_out = helper.make_tensor_value_info(node['in_nodes_name'][0] + 'r', dtype, shape=node['in_tensors'][0])
        onnx_value.append(relu_out)
        relu_node, out = make_node('Relu', [x], [node['in_nodes_name'][0] + 'r'])
        onnx_node.append(relu_node)

        out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], dtype, shape=node['out_tensors'][0])
        onnx_value.append(out_v)

        max_v = np.array(6).astype(node['dtype'])
        max_value = numpy_helper.from_array(max_v, name='max_v')
        onnx_init.append(max_value)
        min_node, out = make_node('Clip', inputs=[out, "", 'max_v'], outputs=node['out_nodes_name'])
        onnx_node.append(min_node)

        return onnx_node, onnx_value, onnx_init


@OpMapper(["PRelu"])
class PRelu():

    @classmethod
    def version_1(cls, node, **kwargs):

        onnx_node = []
        onnx_value = []
        onnx_init = []
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        # get input, output
        x = node['in_nodes_name'][0]
        y = node['out_nodes_name'][0]
        y_shape = node['out_tensors'][0]
        out = helper.make_tensor_value_info(y, dtype, shape=y_shape)
        onnx_value.append(out)
        # get train weights
        slope_v = node['node'].layer.alpha
        slope_n = node['node'].layer.__class__.__name__ + '/alpha'
        weights = numpy_helper.from_array(arr=to_numpy(slope_v), name=slope_n)
        onnx_init.append(weights)
        # make prelu node
        p_node, out = make_node('PRelu', inputs=[x, slope_n], outputs=[y])
        onnx_node.append(p_node)
        return onnx_node, onnx_value, onnx_init


@OpMapper(["Mish"])
class Mish():

    @classmethod
    def version_1(cls, node, **kwargs):

        onnx_node = []
        onnx_value = []
        onnx_init = []
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        # get input, output
        x = node['in_nodes_name'][0]
        y = node['out_nodes_name'][0]
        y_shape = node['out_tensors'][0]
        # make softplus node
        s_value = helper.make_tensor_value_info(y + '_softplus', dtype, y_shape)
        onnx_value.append(s_value)
        s_node, out = make_node('Softplus', inputs=[x], outputs=[y + '_softplus'])
        onnx_node.append(s_node)
        # make tanh node
        t_value = helper.make_tensor_value_info(y + '_tanh', dtype, y_shape)
        onnx_value.append(t_value)
        t_node, out = make_node('Tanh', inputs=[out], outputs=[y + '_tanh'])
        onnx_node.append(t_node)
        # make matmul
        out_v = helper.make_tensor_value_info(y, dtype, shape=y_shape)
        onnx_value.append(out_v)
        o_node, _ = make_node('Mul', inputs=[x, out], outputs=[y])
        onnx_node.append(o_node)
        return onnx_node, onnx_value, onnx_init


@OpMapper(["Swish"])
class Swish():

    @classmethod
    def version_1(cls, node, **kwargs):

        onnx_node = []
        onnx_value = []
        onnx_init = []
        dtype = NP_TYPE_TO_TENSOR_TYPE[node['dtype']]
        # get input, output
        x = node['in_nodes_name'][0]
        y = node['out_nodes_name'][0]
        y_shape = node['out_tensors'][0]
        # make softplus node
        s_value = helper.make_tensor_value_info(y + '_sigmoid', dtype, y_shape)
        onnx_value.append(s_value)
        s_node, out = make_node('Sigmoid', inputs=[x], outputs=[y + '_sigmoid'])
        onnx_node.append(s_node)
        # make matmul
        out_v = helper.make_tensor_value_info(y, dtype, shape=y_shape)
        onnx_value.append(out_v)
        o_node, _ = make_node('Mul', inputs=[x, out], outputs=[y])
        onnx_node.append(o_node)
        return onnx_node, onnx_value, onnx_init