#! /usr/bin/python
# -*- coding: utf-8 -*-

from onnx import helper, TensorProto, numpy_helper
from ..op_mapper import OpMapper
from ...common import make_node, to_numpy
from ..datatype_mapping import NP_TYPE_TO_TENSOR_TYPE
from ...common import tlx_act_2_onnx


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
        y = node['node'].layer.name + '/weights'
        weights = numpy_helper.from_array(arr=to_numpy(node['node'].layer.W), name=y)
        onnx_init.append(weights)

        # Cast x type to float32
        if str(node['dtype']) != 'float32':
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

            if node['node'].layer.act is not None:
                # Build addition
                b_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'b', TensorProto.FLOAT, shape=node['out_tensors'][0])
                onnx_value.append(b_v)
                b = numpy_helper.from_array(arr=to_numpy(node['node'].layer.b), name=node['node'].layer.name + '/b')
                onnx_init.append(b)
                b_node, out = make_node('Add', inputs=[out, node['node'].layer.name + '/b'], outputs=[node['out_nodes_name'][0] + 'b'])
                onnx_node.append(b_node)
                # Build activation
                act_op = node['node'].layer.act.__class__.__name__
                out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT, shape=node['out_tensors'][0])
                onnx_value.append(out_v)
                # Using Opmapper
                act_node, _ = tlx_act_2_onnx[act_op]([out], node['out_nodes_name'], node['node'].layer.act)
                onnx_node.append(act_node)

            else:
                # Build addition
                out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT, shape=node['out_tensors'][0])
                onnx_value.append(out_v)
                b = numpy_helper.from_array(arr=to_numpy(node['node'].layer.b), name=node['node'].layer.name + '/b')
                onnx_init.append(b)
                o_node, _ = make_node('Add', inputs=[out, node['node'].layer.name + '/b'], outputs=node['out_nodes_name'])
                onnx_node.append(o_node)

        else:
            if node['node'].layer.act is not None:
                # Build multiplication
                act_op = node['node'].layer.act.__class__.__name__
                m_v = helper.make_tensor_value_info(node['out_nodes_name'][0] + 'm', TensorProto.FLOAT, shape=node['out_tensors'][0])
                onnx_value.append(m_v)
                m_node, out = make_node('MatMul', inputs=[x, y], outputs=node['out_nodes_name'][0] + 'm')
                onnx_node.append(m_node)
                # Build activation
                out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT, shape=node['out_tensors'][0])
                onnx_value.append(out_v)
                act_node, out = tlx_act_2_onnx[act_op]([out], node['out_nodes_name'], node['node'].layer.act)
                onnx_node.append(act_node)
            else:
                out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], TensorProto.FLOAT, shape=node['out_tensors'][0])
                onnx_value.append(out_v)
                o_node, out = make_node('MatMul', inputs=[x, y], outputs=node['out_nodes_name'])
                onnx_node.append(o_node)


        if str(node['dtype']) != 'float32':
            out_v = helper.make_tensor_value_info(node['out_nodes_name'][0], NP_TYPE_TO_TENSOR_TYPE[node['dtype']],
                                                shape=node['out_tensors'][0])
            onnx_value.append(out_v)
            c_node, out = make_node('Cast', inputs=[out], outputs=node['out_nodes_name'],
                                    to=NP_TYPE_TO_TENSOR_TYPE[node['dtype']])
            onnx_node.append(c_node)

        return onnx_node, onnx_value, onnx_init