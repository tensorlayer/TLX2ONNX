#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto, numpy_helper
from .op_mapper import OpMapper
from ..common import make_node, to_numpy


@OpMapper('Linear')
class Linear():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        y = node['node'].layer.W.name
        weights = numpy_helper.from_array(arr=to_numpy(node['node'].layer.W), name=y)
        onnx_init.append(weights)
        print(to_numpy(node['node'].layer.W).shape)
        if node['dtype'] == 'float64':
            m_out = helper.make_tensor_value_info(node['in_nodes_name'][0] + 'm', TensorProto.FLOAT, shape=node['out_tensors'][0])
            onnx_value.append(m_out)
            mat, m_o = make_node('MatMul', inputs=[x, y], outputs=node['out_nodes_name'][0] + 'm')
            onnx_node.append(mat)

            out = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.DOUBLE, shape=node['out_tensors'][0])
            onnx_value.append(out)
            o_node, _ = make_node('Cast', inputs=m_o, to=TensorProto.DOUBLE, outputs=node['out_nodes_name'])
            onnx_node.append(o_node)
        else:
            out = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT, shape=node['out_tensors'][0])
            onnx_value.append(out)
            o_node, _ = make_node('MatMul', inputs=[x, y], outputs=node['out_nodes_name'])
            onnx_node.append(o_node)

        if node['node'].layer.b_init is not None:
            # TODO Extend to add bias
            pass

        return onnx_node, onnx_value, onnx_init