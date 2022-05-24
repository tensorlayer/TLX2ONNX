#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto
from .op_mapper import OpMapper
from ..common import make_node, transpose_shape


@OpMapper('matmul')
class MatMul():
    # supports v1-v12

    @classmethod
    def version_1(cls, node, **kwargs):
        onnx_node = []
        onnx_value = []
        onnx_init = []

        x = node['in_nodes_name'][0]
        y = node['in_nodes_name'][1]

        if node.attr('transpose_X'):
            perm = list(range(len(node['in_tensors'][0])))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if node['dtype'] == 'float64':
                cxv = helper.make_tensor_value_info(node['in_nodes_name'][0] + '_cast', TensorProto.FLOAT,
                                                   shape=node['in_tensors'][0])
                onnx_value.append(cxv)
                cxn, x = make_node('Cast', inputs=[x], outputs=[node['in_nodes_name'][0] + '_cast'], to=TensorProto.FLOAT)
                onnx_node.append(cxn)

                cxtv = helper.make_tensor_value_info(node['in_nodes_name'][0] + '_t', TensorProto.FLOAT,
                                                     shape=transpose_shape(node['in_tensors'][0], perm))
                onnx_value.append(cxtv)
                cxt, x = make_node('Transpose', inputs=[x], outputs=[['in_nodes_name'][0]] + '_t', perm=perm)
                onnx_node.append(cxt)
            else:
                cxtv = helper.make_tensor_value_info(node['in_nodes_name'][0] + '_t', TensorProto.FLOAT,
                                                     shape=transpose_shape(node['in_tensors'][0], perm))
                onnx_value.append(cxtv)
                cxt, x = make_node('Transpose', inputs=[x], outputs=[['in_nodes_name'][0]] + '_t', perm=perm)
                onnx_node.append(cxt)

        if node.attr('transpose_Y'):
            perm = list(range(len(node['in_tensors'][1])))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if node['dtype'] == 'float64':
                cyv = helper.make_tensor_value_info(node['in_nodes_name'][1] + '_cast', TensorProto.FLOAT,
                                                   shape=node['in_tensors'][1])
                onnx_value.append(cyv)
                cyn, y = make_node('Cast', inputs=[y], outputs=[node['in_nodes_name'][1] + '_cast'], to=TensorProto.FLOAT)
                onnx_node.append(cyn)

                cytv = helper.make_tensor_value_info(node['in_nodes_name'][1] + '_t', TensorProto.FLOAT,
                                                     shape=transpose_shape(node['in_tensors'][1], perm))
                onnx_value.append(cytv)
                cyt, y = make_node('Transpose', inputs=[y], outputs=[node['in_nodes_name'][1] + '_t'], perm=perm)
                onnx_node.append(cyt)
            else:
                cytv = helper.make_tensor_value_info(node['in_nodes_name'][1] + '_t', TensorProto.FLOAT,
                                                     shape=transpose_shape(node['in_tensors'][1], perm))
                onnx_value.append(cytv)
                cyt, y = make_node('Transpose', inputs=y, outputs=[node['in_nodes_name'][1] + '_t'], perm=perm)
                onnx_node.append(cyt)

        if node['dtype'] == 'float64':
            m_out = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'm', TensorProto.FLOAT, shape=node['out_tensors'])
            onnx_value.append(m_out)
            mat, m_o = make_node('MatMul', inputs=[x, y], outputs=node['out_nodes_name'][0] + 'm')
            onnx_node.append(mat)

            out = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.DOUBLE, shape=node['out_tensors'])
            onnx_value.append(out)
            o_node, _ = make_node('Cast', inputs=m_o, to=TensorProto.DOUBLE, outputs=node['out_nodes_name'])
            onnx_node.append(o_node)
        else:
            out = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT, shape=node['out_tensors'])
            onnx_value.append(out)
            o_node, _ = make_node('MatMul', inputs=[x, y], outputs=node['out_nodes_name'])
            onnx_node.append(o_node)
        return onnx_node, onnx_value, onnx_init
