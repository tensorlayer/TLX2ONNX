#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto
import tensorlayerx as tlx
from .datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
import numpy as np
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

        x = helper.make_tensor_value_info(node['in_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                          shape=node['in_tensors'][0])
        onnx_value.append(x)
        y = helper.make_tensor_value_info(node['in_nodes_name'][1], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                          shape=node['in_tensors'][1])
        onnx_value.append(y)

        if node.attr('transpose_X'):
            perm = list(range(len(node['in_tensors'][0])))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if node['dtype'] == 'float64':
                cxv = helper.make_tensor_value_info(node['in_nodes_name'][0] + '_cast', TensorProto.FLOAT,
                                                   shape=node['in_tensors'][0])
                onnx_value.append(cxv)
                cxn = make_node('Cast', inputs=node['in_nodes_name'][0], outputs=node['in_nodes_name'][0] + '_cast', to=TensorProto.FLOAT)
                onnx_node.append(cxn)

            cxtv = helper.make_tensor_value_info(node['in_nodes_name'][0] + '_t', TensorProto.FLOAT,
                                                 shape=transpose_shape(node['in_tensors'][0], perm))
            onnx_value.append(cxtv)
            cxt = make_node('Transpose', inputs=node['in_nodes_name'][0] + '_cast', outputs=['in_nodes_name'][0] + '_t', perm=perm)
            onnx_node.append(cxt)

        if node.attr('transpose_Y'):
            perm = list(range(len(node['in_tensors'][1])))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            if node['dtype'] == 'float64':
                cyv = helper.make_tensor_value_info(node['in_nodes_name'][1] + '_cast', TensorProto.FLOAT,
                                                   shape=node['in_tensors'][1])
                onnx_value.append(cyv)
                cyn = make_node('Cast', inputs=node['in_nodes_name'][1], outputs=node['in_nodes_name'][1] + '_cast', to=TensorProto.FLOAT)
                onnx_node.append(cyn)

            cxtv = helper.make_tensor_value_info(node['in_nodes_name'][1] + '_t', TensorProto.FLOAT,
                                                 shape=transpose_shape(node['in_tensors'][1], perm))
            onnx_value.append(cxtv)
            y = make_node('Transpose', inputs=node['in_nodes_name'][1] + '_cast', outputs=node['in_nodes_name'][1] + '_t', perm=perm)

        if node.attr('alpha') == 1.0:
            if node['dtype'] == 'float64':
                if node.attr('transpose_X') and node.attr('transpose_Y'):
                    out = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT,
                                                        shape=node['out_tensors'][0])
                    onnx_value.append(out)
                    output_node = make_node('MatMul',
                                            inputs=[node['in_nodes_name'][0] + '_t', node['in_nodes_name'][1] + '_t'],
                                            outputs=node['out_nodes_name'][0])
                    onnx_node.append(output_node)
                elif not node.attr('transpose_X') and node.attr('transpose_Y'):
                    pass

        #         output_node = make_node('MatMul', inputs=[x, y])
        #         graph.make_node(
        #             'Cast',
        #             inputs=output_node,
        #             to=dtypes.ONNX.DOUBLE,
        #             outputs=node.output('Out'))
        #     else:
        #         graph.make_node(
        #             'MatMul', inputs=[x, y], outputs=node.output('Out'))
        # else:
        #     if node.input_dtype('X', 0) == paddle.float64:
        #         output_node = graph.make_node('MatMul', inputs=[x, y])
        #         matmul = graph.make_node(
        #             'Cast', inputs=output_node, to=dtypes.ONNX.DOUBLE)
        #         scale = graph.make_node(
        #             'Constant',
        #             dtype=dtypes.ONNX.DOUBLE,
        #             value=node.attr('alpha'))
        #     else:
        #         matmul = graph.make_node('MatMul', inputs=[x, y])
        #         scale = graph.make_node(
        #             'Constant',
        #             dtype=dtypes.ONNX.FLOAT,
        #             value=node.attr('alpha'))
        #
        #     onnx_node = graph.make_node(
        #         'Mul', inputs=[matmul, scale], outputs=node.output('Out'))