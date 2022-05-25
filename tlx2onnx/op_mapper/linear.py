#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto, numpy_helper
from .op_mapper import OpMapper
from ..common import make_node, to_numpy
from .datatype_mapping import NP_TYPE_TO_TENSOR_TYPE


@OpMapper('Linear')
class Linear():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        # TODO How to compatible multiple framework parameter names
        y = node['node'].layer.W.name
        weights = numpy_helper.from_array(arr=to_numpy(node['node'].layer.W), name=y)
        onnx_init.append(weights)

        # Cast x type to float32
        if node['dtype'] != 'float32':
            c_x = helper.make_tensor_value_info(node['in_nodes_name'][0] + 'c', TensorProto.FLOAT, shape=node['in_tensors'][0])
            onnx_value.append(c_x)
            c_node, x = make_node('Cast', inputs=[x], outputs=[node['in_nodes_name'][0] + 'c'], to=TensorProto.FLOAT)
            onnx_node.append(c_node)

        if node['node'].layer.b_init is not None:
            # Build multiplication
            m_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'm', TensorProto.FLOAT, shape=node['out_tensors'][0])
            onnx_value.append(m_v)
            m_node, out = make_node('MatMul', inputs=[x, y], outputs=[node['out_nodes_name'][0] + 'm'])
            onnx_node.append(m_node)
            # Build addition
            out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT, shape=node['out_tensors'][0])
            onnx_value.append(out_v)
            b = numpy_helper.from_array(arr=to_numpy(node['node'].layer.b), name=node['node'].layer.b.name)
            onnx_init.append(b)
            o_node, out = make_node('Add', inputs=[out, node['node'].layer.b.name], outputs=node['out_nodes_name'])
            onnx_node.append(o_node)
        else:
            out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT, shape=node['out_tensors'][0])
            onnx_value.append(out_v)
            o_node, out = make_node('MatMul', inputs=[x, y], outputs=node['out_nodes_name'])
            onnx_node.append(o_node)

        if node['dtype'] != 'float32':
            out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dytpe']],
                                                shape=node['out_tensors'][0])
            onnx_value.append(out_v)
            c_node, _ = make_node('Cast', inputs=[out], outputs=node['out_nodes_name'], to=TensorProto.FLOAT)
            onnx_node.append(c_node)

        return onnx_node, onnx_value, onnx_init